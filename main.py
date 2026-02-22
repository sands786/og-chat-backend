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

@app.on_event("startup")
async def startup():
    og.init(
        private_key=os.environ.get("OG_PRIVATE_KEY"),
        email=os.environ.get("OG_EMAIL", ""),
        password=os.environ.get("OG_PASSWORD", ""),
    )

@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        chat_output, tx_hash, _ = og.llm_chat(
            model_cid=req.model,
            messages=messages,
            max_tokens=512,
            temperature=0.7,
        )
        return {
            "content": chat_output,
            "payment_hash": tx_hash,
            "model": req.model,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "ok"}
