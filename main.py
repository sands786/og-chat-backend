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
    client = og.new_client(private_key=private_key, email=None, password=None)

    try:
        result = client.llm_chat(
            model_cid="openai/gpt-4o",
            messages=messages,
            max_tokens=512,
            temperature=0.7,
            inference_mode=og.LlmInferenceMode.TEE,
        )
        return {
            "content": result.chat_output["content"],
            "payment_hash": str(result.payment_hash),
            "model": "gpt-4o-tee",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
