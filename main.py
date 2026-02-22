from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import opengradient as og
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    model: str = "openai/gpt-4o-mini"

@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        private_key = os.environ.get("OG_PRIVATE_KEY")
        if not private_key:
            raise HTTPException(status_code=500, detail="OG_PRIVATE_KEY not configured")

        client = og.new_client(private_key=private_key)

        messages = [{"role": m.role, "content": m.content} for m in req.messages]

        result = client.llm_chat(
            model_cid=req.model,
            messages=messages,
            max_tokens=512,
            temperature=0.7,
        )

        return {
            "content": result.chat_output,
            "payment_hash": result.payment_hash,
            "model": req.model,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "ok", "message": "OG Chat Backend running!"}
