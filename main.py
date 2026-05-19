"""
Voice Assistant Backend — FastAPI + WebSocket
=============================================
Deploy on Render as a Python web service.

Flow:
  1. Client sends audio (webm/wav/m4a) via WebSocket or REST
  2. Server transcribes via Groq Whisper
  3. Streams LLM tokens (Groq Llama)
  4. Synthesises each sentence with Edge-TTS
  5. Streams audio chunks back to client in real-time

WebSocket message protocol (JSON):
  Client → Server:  { "type": "audio", "data": "<base64 audio>", "format": "webm" }
  Client → Server:  { "type": "clear_history" }
  Server → Client:  { "type": "transcript",   "text": "..." }
  Server → Client:  { "type": "text_chunk",   "text": "..." }
  Server → Client:  { "type": "audio_chunk",  "data": "<base64 mp3>", "is_last": false }
  Server → Client:  { "type": "done" }
  Server → Client:  { "type": "error",        "message": "..." }
"""

import asyncio
import base64
import io
import os
import re
import uuid
import wave
from contextlib import asynccontextmanager
from typing import AsyncIterator

import edge_tts
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from groq import AsyncGroq
import uvicorn

# ════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
WHISPER_MODEL = "whisper-large-v3-turbo"   # fastest
LLM_MODEL     = "llama-3.3-70b-versatile"
TTS_VOICE     = "hi-IN-NeerjaNeural"        # natural Hindi female
TTS_RATE      = "+8%"
TTS_PITCH     = "0Hz"

SYSTEM_PROMPT = (
    "Tum ek helpful, friendly aur natural female assistant ho. "
    "Tum Hindi mein baat karte ho lekin agar user English ya Hinglish mein bole "
    "to wahi language use karo. "
    "Jawab hamesha chhote, crisp aur conversational rakho — "
    "jaise ek dost baat kar raha ho. "
    "Koi bullet list ya markdown mat use karo, sirf plain baat karo."
)

# ════════════════════════════════════════════════════════════
# APP
# ════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.groq = AsyncGroq(api_key=GROQ_API_KEY)
    yield
    await app.state.groq.close()

app = FastAPI(title="Voice Assistant API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════

def split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries. Merge very short fragments."""
    parts = re.split(r"(?<=[।.?!\n])\s*", text.strip())
    merged, buf = [], ""
    for p in parts:
        buf = (buf + " " + p).strip()
        if len(buf) >= 12:
            merged.append(buf)
            buf = ""
    if buf:
        merged.append(buf)
    return merged or [text]


async def transcribe(client: AsyncGroq, audio_bytes: bytes, fmt: str = "webm") -> str:
    """Send audio bytes to Groq Whisper. Auto-detects language."""
    mime = {
        "webm": "audio/webm",
        "wav":  "audio/wav",
        "mp4":  "audio/mp4",
        "m4a":  "audio/mp4",
        "ogg":  "audio/ogg",
    }.get(fmt, "audio/webm")

    result = await client.audio.transcriptions.create(
        file=(f"audio.{fmt}", audio_bytes, mime),
        model=WHISPER_MODEL,
        language=None,          # auto-detect: Hindi / English / Hinglish
        response_format="text",
    )
    return str(result).strip()


async def synthesise_sentence(text: str) -> bytes:
    """Convert one sentence to MP3 bytes via Edge-TTS."""
    communicate = edge_tts.Communicate(text, TTS_VOICE, rate=TTS_RATE, pitch=TTS_PITCH)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()


async def stream_llm_sentences(
    client: AsyncGroq,
    history: list[dict],
    user_msg: str,
) -> AsyncIterator[tuple[str, str]]:
    """
    Stream LLM response sentence by sentence.
    Yields (sentence_text, full_response_so_far).
    Also appends assistant turn to history in-place.
    """
    history.append({"role": "user", "content": user_msg})

    stream = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
        stream=True,
        max_tokens=400,
        temperature=0.75,
    )

    buffer       = ""
    full_response = ""

    async for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        buffer        += token
        full_response += token

        sentences = split_sentences(buffer)
        if len(sentences) > 1:
            for s in sentences[:-1]:
                if s.strip():
                    yield s.strip(), full_response
            buffer = sentences[-1]

    if buffer.strip():
        yield buffer.strip(), full_response

    history.append({"role": "assistant", "content": full_response})


# ════════════════════════════════════════════════════════════
# WEBSOCKET  — main real-time endpoint
# ════════════════════════════════════════════════════════════

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    client  = app.state.groq
    history: list[dict] = []

    # Send greeting audio on connect
    greeting = "Namaste! Main aapki assistant hoon. Batao, kya madad kar sakti hoon?"
    try:
        g_audio = await synthesise_sentence(greeting)
        await ws.send_json({
            "type":    "audio_chunk",
            "data":    base64.b64encode(g_audio).decode(),
            "text":    greeting,
            "is_last": True,
        })
        await ws.send_json({"type": "done"})
    except Exception:
        pass

    try:
        while True:
            msg = await ws.receive_json()

            # ── Clear history ──
            if msg.get("type") == "clear_history":
                history.clear()
                await ws.send_json({"type": "done"})
                continue

            # ── Audio input ──
            if msg.get("type") != "audio":
                continue

            audio_b64 = msg.get("data", "")
            fmt       = msg.get("format", "webm")

            if not audio_b64:
                await ws.send_json({"type": "error", "message": "No audio data"})
                continue

            try:
                audio_bytes = base64.b64decode(audio_b64)
            except Exception:
                await ws.send_json({"type": "error", "message": "Invalid base64"})
                continue

            # 1. Transcribe
            try:
                transcript = await transcribe(client, audio_bytes, fmt)
            except Exception as e:
                await ws.send_json({"type": "error", "message": f"STT error: {e}"})
                continue

            if not transcript:
                await ws.send_json({"type": "error", "message": "Could not understand audio"})
                continue

            await ws.send_json({"type": "transcript", "text": transcript})

            # 2. Stream LLM → synthesise each sentence → send audio chunk
            try:
                async for sentence, _ in stream_llm_sentences(client, history, transcript):
                    # Send text chunk immediately (for subtitles / UI)
                    await ws.send_json({"type": "text_chunk", "text": sentence})

                    # Synthesise and send audio
                    audio = await synthesise_sentence(sentence)
                    await ws.send_json({
                        "type":    "audio_chunk",
                        "data":    base64.b64encode(audio).decode(),
                        "text":    sentence,
                        "is_last": False,
                    })

                await ws.send_json({"type": "done"})

            except Exception as e:
                await ws.send_json({"type": "error", "message": f"Pipeline error: {e}"})

    except WebSocketDisconnect:
        pass


# ════════════════════════════════════════════════════════════
# REST FALLBACK  — for simple HTTP clients / testing
# ════════════════════════════════════════════════════════════

@app.post("/chat/audio")
async def chat_audio(
    audio: UploadFile = File(...),
    session_id: str   = Form(default="default"),
):
    """
    Upload audio file → get back JSON with transcript + full TTS audio (base64).
    Stateless (no history). For quick tests.
    """
    client      = app.state.groq
    audio_bytes = await audio.read()
    fmt         = audio.filename.rsplit(".", 1)[-1] if audio.filename else "webm"

    # STT
    try:
        transcript = await transcribe(client, audio_bytes, fmt)
    except Exception as e:
        return JSONResponse({"error": f"STT failed: {e}"}, status_code=500)

    # LLM (non-streaming for REST)
    completion = await client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": transcript},
        ],
        max_tokens=400,
        temperature=0.75,
    )
    response_text = completion.choices[0].message.content.strip()

    # TTS
    try:
        audio_out = await synthesise_sentence(response_text)
        audio_b64 = base64.b64encode(audio_out).decode()
    except Exception as e:
        audio_b64 = None

    return JSONResponse({
        "transcript":   transcript,
        "response":     response_text,
        "audio_base64": audio_b64,   # MP3 base64
        "audio_format": "mp3",
    })


@app.get("/health")
async def health():
    return {"status": "ok", "model": LLM_MODEL, "tts_voice": TTS_VOICE}


# ════════════════════════════════════════════════════════════
# LOCAL RUN
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)