from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import opengradient as og
import os
import uvicorn

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

        # Get content from response
        content = ""
        if hasattr(result, 'chat_output'):
            if isinstance(result.chat_output, dict):
                content = result.chat_output.get("content", str(result.chat_output))
            else:
                content = str(result.chat_output)
        elif hasattr(result, 'content'):
            content = result.content
        else:
            content = str(result)

        # Get tx hash — try multiple possible field names
        tx_hash = ""
        for field in ['transaction_hash', 'payment_hash', 'tx_hash', 'txHash']:
            val = getattr(result, field, None)
            if val:
                tx_hash = str(val)
                break

        print(f"SUCCESS - tx_hash: {tx_hash}")
        print(f"Content preview: {content[:100]}")

        return {
            "content": content,
            "payment_hash": tx_hash,
            "model": "gpt-4o-tee",
        }

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")

@app.get("/")
def root():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
