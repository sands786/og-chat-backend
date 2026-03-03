from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import opengradient as og
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
    images: Optional[List[ImageData]] = None  # ← NEW: optional images field

class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "openai/gpt-4o"

# ── Helper: build OpenAI-compatible messages ─────────────────────

def build_messages(messages: List[Message]):
    """
    Converts our Message objects into the format OpenAI expects.
    - Text-only messages → {"role": ..., "content": "string"}
    - Messages with images → {"role": ..., "content": [text_block, image_block, ...]}
    """
    result = []
    for m in messages:
        if m.images:
            # Vision message — build content as array of blocks
            content_blocks = []

            # Add the text part first (if any)
            if m.content and m.content.strip():
                content_blocks.append({
                    "type": "text",
                    "text": m.content
                })

            # Add each image as a base64 data URL
            for img in m.images:
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{img.type};base64,{img.data}",
                        "detail": "auto"   # let OpenAI pick low/high res automatically
                    }
                })

            result.append({"role": m.role, "content": content_blocks})
        else:
            # Plain text message
            result.append({"role": m.role, "content": m.content})

    return result

# ── Chat endpoint ────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(req: ChatRequest):
    private_key = os.environ.get("OG_PRIVATE_KEY")
    if not private_key:
        raise HTTPException(status_code=500, detail="OG_PRIVATE_KEY not set")

    # Build messages in OpenAI vision-compatible format
    messages = build_messages(req.messages)

    # Detect if this request contains images (for logging / model override)
    has_images = any(m.images for m in req.messages if m.images)

    try:
        client = og.Client(private_key=private_key)

        result = client.llm.chat(
            model="openai/gpt-4o",   # GPT-4o supports vision natively
            messages=messages,
            max_tokens=1024,          # bumped up — vision responses tend to be longer
            temperature=0.7,
        )

        tee_signature  = getattr(result, 'tee_signature', None)
        tee_timestamp  = getattr(result, 'tee_timestamp', None)

        tee_time_formatted = ""
        if tee_timestamp:
            try:
                tee_time_formatted = datetime.utcfromtimestamp(tee_timestamp).strftime('%Y-%m-%d %H:%M:%S UTC')
            except Exception:
                tee_time_formatted = str(tee_timestamp)

        return {
            "content":       result.chat_output["content"],
            "tee_signature": tee_signature,
            "tee_timestamp": tee_time_formatted,
            "model":         "gpt-4o-tee",
            "has_vision":    has_images,   # optional flag for frontend
        }

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ── Health check ─────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
