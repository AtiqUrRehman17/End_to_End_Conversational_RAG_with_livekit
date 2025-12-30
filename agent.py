import os
import httpx
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import (
    AgentServer,
    AgentSession,
    Agent,
    room_io,
    function_tool
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env")

RAG_API_BASE = "http://localhost:8000"


# =========================
# TOOLS
# =========================

@function_tool(
    name="check_document_status",
    description="Check whether a PDF document has been uploaded"
)
async def check_document_status_tool() -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(f"{RAG_API_BASE}/health")

    if response.status_code != 200:
        return "Unable to check document status."

    data = response.json()
    return (
        "Document is uploaded and ready."
        if data.get("chatbot_initialized")
        else "No document uploaded."
    )


@function_tool(
    name="upload_pdf",
    description="Upload a PDF file so the assistant can answer questions from it"
)
async def upload_pdf_tool(file_path: str) -> str:
    if not os.path.exists(file_path):
        return "The file path does not exist."

    async with httpx.AsyncClient(timeout=120) as client:
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f, "application/pdf")}
            response = await client.post(f"{RAG_API_BASE}/upload-pdf", files=files)

    if response.status_code != 200:
        return f"Failed to upload PDF: {response.text}"

    return "The PDF has been uploaded successfully. I am now ready to answer questions from it."


@function_tool(
    name="query_pdf",
    description="Ask a question about the uploaded PDF document"
)
async def query_pdf_tool(question: str) -> str:
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(
            f"{RAG_API_BASE}/query",
            json={"query": question}
        )

    if response.status_code != 200:
        return f"Failed to query the document: {response.text}"

    return response.json()["answer"]


# =========================
# AGENT
# =========================

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are a helpful voice assistant.

You can:
- Have normal conversations
- Help users upload a PDF
- Answer questions from an uploaded PDF

IMPORTANT RULES:
1. If the user asks about a document, FIRST check document status.
2. If no document exists, tell the user to upload one.
3. If a document exists, ALWAYS use the query_pdf tool.
4. Never answer document questions from general knowledge.
5. Be concise and natural.
""",
            tools=[
                check_document_status_tool,
                upload_pdf_tool,
                query_pdf_tool,
            ],
        )


# =========================
# LIVEKIT SESSION
# =========================

server = AgentServer()


@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    session = AgentSession(
        stt="assemblyai/universal-streaming:en",
        llm="openai/gpt-4.1-mini",
        tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind
                    == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )

    await session.generate_reply(
        instructions=(
            "Hello. You can chat with me normally, "
            "or upload a PDF and ask questions from it."
        )
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
