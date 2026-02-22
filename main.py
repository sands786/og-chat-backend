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
    model: str = "GPT_4O_MINI"

# Map model strings to og.TEE_LLM enum values
MODEL_MAP = {
    "openai/gpt-4o-mini": og.TEE_LLM.GPT_4O_MINI,
    "openai/gpt-4o": og.TEE_LLM.GPT_4O,
    "meta-llama/Meta-Llama-3-8B-Instruct": og.TEE_LLM.LLAMA_3_8B,
    "mistralai/Mistral-7B-Instruct-v0.3": og.TEE_LLM.MISTRAL_7B,
}

@app.on_event("startup")
async def startup():
    private_key = os.environ.get("OG_PRIVATE_KEY")
    if not private_key:
        raise RuntimeError("OG_PRIVATE_KEY not set")
    app.state.client = og.Client(private_key=private_key)
    # Ensure wallet has OPG approved for spending
    app.state.client.llm.ensure_opg_approval(opg_amount=5.0)

@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        client = app.state.client
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        
        model_enum = MODEL_MAP.get(req.model, og.TEE_LLM.GPT_4O_MINI)
        
        result = client.llm.chat(
            model=model_enum,
            messages=messages,
            max_tokens=512,
            temperature=0.7,
        )

        return {
            "content": result.chat_output["content"],
            "payment_hash": result.payment_hash,
            "model": req.model,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "ok"}
