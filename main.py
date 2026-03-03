from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
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

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "openai/gpt-4o"

@app.post("/api/chat")
async def chat(req: ChatRequest):
    private_key = os.environ.get("OG_PRIVATE_KEY")
    if not private_key:
        raise HTTPException(status_code=500, detail="OG_PRIVATE_KEY not set")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    try:
        client = og.Client(private_key=private_key)
        result = client.llm.chat(
            model="openai/gpt-4o",
            messages=messages,
            max_tokens=512,
            temperature=0.7,
        )

        content = result.chat_output["content"]
        tee_signature = getattr(result, 'tee_signature', None)
        tee_timestamp = getattr(result, 'tee_timestamp', None)

        # Format timestamp
        verified_at = ""
        if tee_timestamp:
            try:
                verified_at = datetime.utcfromtimestamp(tee_timestamp).strftime('%Y-%m-%d %H:%M:%S UTC')
            except:
                verified_at = str(tee_timestamp)

        print(f"TEE SIGNATURE: {tee_signature[:40] if tee_signature else 'None'}...")
        print(f"TEE TIMESTAMP: {verified_at}")

        return {
            "content": content,
            "tee_signature": tee_signature,
            "tee_timestamp": verified_at,
            "payment_hash": None,
            "model": "gpt-4o-tee",
        }

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
