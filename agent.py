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
# SAFE HTTP CALLS
# =========================
async def safe_call(coro_fn, fallback: str):
    try:
        return await asyncio.wait_for(coro_fn(), timeout=60)
    except Exception:
        return fallback


# =========================
# TOOLS
# =========================
@function_tool(
    name="check_document_status",
    description="Check whether a document has been uploaded"
)
async def check_document_status_tool() -> str:
    async def call():
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{RAG_API_BASE}/health")
            data = r.json()
            return (
                "Document is uploaded and ready."
                if data.get("chatbot_initialized")
                else "No document uploaded."
            )
    return await safe_call(call, "Unable to check document status.")


@function_tool(
    name="upload_pdf",
    description="Upload a PDF file"
)
async def upload_pdf_tool(file_path: str) -> str:
    if not os.path.exists(file_path):
        return "The file path does not exist."

    async def call():
        async with httpx.AsyncClient(timeout=120) as client:
            with open(file_path, "rb") as f:
                files = {"file": (os.path.basename(file_path), f)}
                await client.post(f"{RAG_API_BASE}/upload-pdf", files=files)
        return "PDF uploaded successfully."
    return await safe_call(call, "Failed to upload PDF.")


@function_tool(
    name="query_pdf",
    description="Query the uploaded document"
)
async def query_pdf_tool(question: str) -> str:
    async def call():
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{RAG_API_BASE}/query",
                json={"query": question}
            )
            return r.json()["answer"]
    return await safe_call(call, "No answer found in the document.")


# =========================
# AGENT
# =========================
class Assistant(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
You are a friendly professional assistant.

- Answer general questions normally.
- If the user asks about a document:
  ALWAYS call query_pdf.
  Speak EXACTLY what query_pdf returns.
""",
            tools=[
                check_document_status_tool,
                upload_pdf_tool,
                query_pdf_tool,
            ]
        )


# =========================
# LIVEKIT SESSION
# =========================
server = AgentServer()

# Helper to split long text for TTS
def chunk_text(text, max_chars=400):
    words = text.split()
    chunks = []
    current = []
    count = 0
    for w in words:
        count += len(w) + 1
        if count > max_chars:
            chunks.append(" ".join(current))
            current = [w]
            count = len(w) + 1
        else:
            current.append(w)
    if current:
        chunks.append(" ".join(current))
    return chunks


@server.rtc_session()
async def my_agent(ctx: agents.JobContext):
    session = AgentSession(
        stt="deepgram/nova-2:en",
        llm="openai/gpt-4.1-mini",
        tts="elevenlabs/eleven_turbo_v2_5:cgSgspJ2msm6clMCkdW9",
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        allow_interruptions=True,
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

    # Initial greeting
    await session.generate_reply(
        instructions=(
            "Hello sir 😊 I am here to help you. "
            "You can ask general questions or upload documents "
            "like PDF, TXT, DOC, or DOCX."
        )
    )

    # Watchdog (simple text-only for version safety)
    async def watchdog():
        while True:
            await asyncio.sleep(10)
            await session.generate_reply(instructions="I am listening. Please continue.")

    asyncio.create_task(watchdog())

    # Override session TTS to chunk long text
    original_generate_reply = session.generate_reply

    async def safe_generate_reply(*args, **kwargs):
        # Get the instructions text
        text = kwargs.get("instructions", "")
        if len(text) > 400:
            chunks = chunk_text(text, max_chars=400)
            for c in chunks:
                await original_generate_reply(instructions=c)
        else:
            await original_generate_reply(*args, **kwargs)

    session.generate_reply = safe_generate_reply  # monkey-patch


if __name__ == "__main__":
    agents.cli.run_app(server)
