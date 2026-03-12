from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import uvicorn
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic Models ──────────────────────────────────────────────

class ImageData(BaseModel):
    type: str   # e.g. "image/png"
    data: str   # raw base64 string (no data URL prefix)

class Message(BaseModel):
    role: str
    content: str
    images: Optional[List[ImageData]] = None

class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "openai/gpt-4o"

# ── Helper: build OpenAI-compatible messages ─────────────────────

def build_messages(messages: List[Message]):
    result = []
    for m in messages:
        if m.images:
            content_blocks = []
            if m.content and m.content.strip():
                content_blocks.append({"type": "text", "text": m.content})
            for img in m.images:
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img.type};base64,{img.data}",
                        "detail": "auto"
                    }
                })
            result.append({"role": m.role, "content": content_blocks})
        else:
            result.append({"role": m.role, "content": m.content})
    return result

# ── Attempt 1: OG TEE mode ───────────────────────────────────────

def try_og_tee(client, messages):
    import opengradient as og
    result = client.llm.chat(
        model="openai/gpt-4o",
        messages=messages,
        max_tokens=1024,
        temperature=0.7,
        inference_mode=og.LlmInferenceMode.TEE,
    )
    return result, "TEE"

# ── Attempt 2: OG VANILLA mode ──────────────────────────────────

def try_og_vanilla(client, messages):
    import opengradient as og
    result = client.llm.chat(
        model="openai/gpt-4o",
        messages=messages,
        max_tokens=1024,
        temperature=0.7,
        inference_mode=og.LlmInferenceMode.VANILLA,
    )
    return result, "VANILLA"

# ── Attempt 3: Direct OpenAI fallback ───────────────────────────

def try_openai_direct(messages_raw):
    """
    Falls back to direct OpenAI API if OG infrastructure is down.
    Requires OPENAI_API_KEY env var.
    """
    import httpx, json

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set — cannot fall back to direct OpenAI")

    payload = {
        "model": "gpt-4o",
        "messages": messages_raw,
        "max_tokens": 1024,
        "temperature": 0.7,
    }

    with httpx.Client(timeout=60) as client:
        resp = client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]

    class FakeResult:
        def __init__(self, text):
            self.chat_output = {"content": text}
            self.tee_signature = None
            self.tee_timestamp = None

    return FakeResult(content), "OPENAI_DIRECT (OG nodes unavailable)"

# ── Chat endpoint ────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(req: ChatRequest):
    private_key = os.environ.get("OG_PRIVATE_KEY")
    if not private_key:
        raise HTTPException(status_code=500, detail="OG_PRIVATE_KEY environment variable is not set")

    messages = build_messages(req.messages)
    has_images = any(m.images for m in req.messages if m.images)

    result = None
    used_mode = "unknown"
    errors = []

    # ── Try OG SDK ───────────────────────────────────────────────
    try:
        import opengradient as og
        client = og.Client(private_key=private_key)

        # Attempt 1: TEE
        try:
            result, used_mode = try_og_tee(client, messages)
            print(f"✓ OG TEE succeeded")
        except Exception as e1:
            errors.append(f"TEE failed: {type(e1).__name__}: {e1}")
            print(f"✗ TEE failed: {e1}")

            # Attempt 2: VANILLA
            try:
                result, used_mode = try_og_vanilla(client, messages)
                print(f"✓ OG VANILLA succeeded")
            except Exception as e2:
                errors.append(f"VANILLA failed: {type(e2).__name__}: {e2}")
                print(f"✗ VANILLA failed: {e2}")

    except ImportError:
        errors.append("opengradient SDK not installed")
        print("✗ opengradient not importable")

    # ── Attempt 3: Direct OpenAI fallback ────────────────────────
    if result is None:
        print(f"OG fully down, trying direct OpenAI... Errors so far: {errors}")
        try:
            result, used_mode = try_openai_direct(messages)
            print(f"✓ OpenAI direct fallback succeeded")
        except Exception as e3:
            errors.append(f"OpenAI direct failed: {type(e3).__name__}: {e3}")
            print(f"✗ OpenAI direct failed: {e3}")

    # ── All attempts failed ──────────────────────────────────────
    if result is None:
        detail = "All inference backends failed. " + " | ".join(errors)
        print(f"ERROR: {detail}")
        raise HTTPException(status_code=503, detail=detail)

    # ── Format response ──────────────────────────────────────────
    tee_signature     = getattr(result, 'tee_signature', None)
    tee_timestamp_raw = getattr(result, 'tee_timestamp', None)
    tee_time_str      = ""
    if tee_timestamp_raw:
        try:
            tee_time_str = datetime.utcfromtimestamp(tee_timestamp_raw).strftime('%Y-%m-%d %H:%M:%S UTC')
        except Exception:
            tee_time_str = str(tee_timestamp_raw)

    return {
        "content":       result.chat_output["content"],
        "tee_signature": tee_signature,
        "tee_timestamp": tee_time_str,
        "model":         "gpt-4o",
        "has_vision":    has_images,
        "used_mode":     used_mode,
    }

# ── Health check ─────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "ok",
        "og_key_set": bool(os.environ.get("OG_PRIVATE_KEY")),
        "openai_key_set": bool(os.environ.get("OPENAI_API_KEY")),
    }

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
