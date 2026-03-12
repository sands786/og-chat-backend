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
    type: str
    data: str

class Message(BaseModel):
    role: str
    content: str
    images: Optional[List[ImageData]] = None

class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "openai/gpt-4o"

# ── Build OpenAI-compatible messages ─────────────────────────────

def build_messages(messages: List[Message]):
    result = []
    for m in messages:
        if m.images:
            blocks = []
            if m.content and m.content.strip():
                blocks.append({"type": "text", "text": m.content})
            for img in m.images:
                blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{img.type};base64,{img.data}", "detail": "auto"}
                })
            result.append({"role": m.role, "content": blocks})
        else:
            result.append({"role": m.role, "content": m.content})
    return result

# ── Attempt 1: OG SDK (no inference_mode — works on all versions) ─

def try_og(messages):
    import opengradient as og
    private_key = os.environ.get("OG_PRIVATE_KEY")
    client = og.Client(private_key=private_key)
    result = client.llm.chat(
        model="openai/gpt-4o",
        messages=messages,
        max_tokens=1024,
        temperature=0.7,
    )
    return result, "OG_TEE"

# ── Attempt 2: Direct OpenAI API fallback ────────────────────────

def try_openai_direct(messages):
    import httpx
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    with httpx.Client(timeout=60) as http:
        resp = http.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "gpt-4o", "messages": messages, "max_tokens": 1024, "temperature": 0.7},
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]

    class _R:
        chat_output = {"content": content}
        tee_signature = None
        tee_timestamp = None

    return _R(), "OPENAI_DIRECT (OG nodes unavailable)"

# ── Chat endpoint ────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not os.environ.get("OG_PRIVATE_KEY"):
        raise HTTPException(status_code=500, detail="OG_PRIVATE_KEY environment variable is not set")

    messages  = build_messages(req.messages)
    has_images = any(m.images for m in req.messages if m.images)

    result    = None
    used_mode = "unknown"
    errors    = []

    # Try OG SDK first
    try:
        result, used_mode = try_og(messages)
        print(f"✓ OG succeeded — mode: {used_mode}")
    except Exception as e1:
        errors.append(f"OG failed: {type(e1).__name__}: {e1}")
        print(f"✗ OG failed: {e1}")

        # Fall back to direct OpenAI
        try:
            result, used_mode = try_openai_direct(messages)
            print(f"✓ OpenAI direct fallback succeeded")
        except Exception as e2:
            errors.append(f"OpenAI fallback failed: {type(e2).__name__}: {e2}")
            print(f"✗ OpenAI fallback failed: {e2}")

    if result is None:
        raise HTTPException(status_code=503, detail="All inference backends failed: " + " | ".join(errors))

    tee_sig = getattr(result, 'tee_signature', None)
    tee_ts  = getattr(result, 'tee_timestamp', None)
    tee_str = ""
    if tee_ts:
        try:    tee_str = datetime.utcfromtimestamp(tee_ts).strftime('%Y-%m-%d %H:%M:%S UTC')
        except: tee_str = str(tee_ts)

    return {
        "content":       result.chat_output["content"],
        "tee_signature": tee_sig,
        "tee_timestamp": tee_str,
        "model":         "gpt-4o",
        "has_vision":    has_images,
        "used_mode":     used_mode,
    }

# ── Health ───────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "status": "ok",
        "og_key_set":     bool(os.environ.get("OG_PRIVATE_KEY")),
        "openai_key_set": bool(os.environ.get("OPENAI_API_KEY")),
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
