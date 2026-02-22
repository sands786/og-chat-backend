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

@app.get("/debug")
def debug():
    private_key = os.environ.get("OG_PRIVATE_KEY")
    try:
        client = og.Client(private_key=private_key)
        methods = [m for m in dir(client) if not m.startswith("_")]
        return {"client_methods": methods}
    except Exception as e:
        og_attrs = [m for m in dir(og) if not m.startswith("_")]
        return {"og_module_attrs": og_attrs, "client_error": str(e)}

@app.get("/")
def root():
    return {"status": "ok"}
```

Commit, wait for Railway to go green, then open:
```
https://og-chat-backend-production.up.railway.app/debug
