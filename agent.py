import os
import httpx
import asyncio
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

    return (
        "The PDF has been uploaded successfully. "
        "I am now ready to answer questions strictly from this document."
    )


@function_tool(
    name="query_pdf",
    description="Ask a question about the uploaded PDF document and return the exact answer from it"
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
You are a friendly, professional voice assistant designed to help users in multiple ways.

GENERAL MODE:
- You can chat normally and answer general questions.
- Be polite, welcoming, and conversational.
- Clearly explain that you can help with questions, guidance, and documents.

DOCUMENT MODE (VERY IMPORTANT):
- Users may upload documents such as PDF, TXT, DOC, or DOCX files.
- When the user asks anything related to an uploaded document:
  1. First check whether a document exists.
  2. If it exists, ALWAYS call query_pdf.
  3. The output of query_pdf IS THE FINAL ANSWER.

CRITICAL RULES:
- NEVER rephrase, summarize, or expand the output of query_pdf.
- Speak the query_pdf result EXACTLY as returned.
- Do NOT add explanations, context, or opinions.
- Do NOT answer document-related questions from your own knowledge.
- If the document does not contain the answer, say exactly what the tool returns.

If no document exists and the user asks about a document:
- Politely ask the user to upload a document first.

Always speak naturally, confidently, and in a friendly manner.
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
        stt="deepgram/nova-2:en",
        llm="openai/gpt-4.1-mini",
        tts="elevenlabs/eleven_turbo_v2_5:cgSgspJ2msm6clMCkdW9",
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        allow_interruptions=True,   # ✅ THIS IS THE KEY FIX
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

    # Initial greeting (NO asyncio.create_task)
    session.generate_reply(
        instructions=(
            "Hello sir, I am here to help you 😊 "
            "You can ask me any general questions, "
            "or I can help you work with files such as PDF, TXT, DOC, or DOCX. "
            "Please let me know how I can assist you today."
        )
    )



if __name__ == "__main__":
    agents.cli.run_app(server)
