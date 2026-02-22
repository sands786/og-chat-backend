from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import opengradient as og
import os
import inspect

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

@app.get("/debug")
def debug():
    private_key = os.environ.get("OG_PRIVATE_KEY")
    try:
        client = og.Client(private_key=private_key)
        methods = [m for m in dir(client) if not m.startswith("_")]
        return {"methods": methods}
    except Exception as e:
        og_methods = [m for m in dir(og) if not m.startswith("_")]
        return {"og_module_methods": og_methods, "client_error": str(e)}

@app.post("/api/chat")
async def chat(req: ChatRequest):
    raise HTTPException(status_code=500, detail="Check /debug endpoint first")

@app.get("/")
def root():
    return {"status": "ok"}
```

Commit it, wait for Railway to redeploy, then open this URL in your browser:
```
https://og-chat-backend-production.up.railway.app/debug
