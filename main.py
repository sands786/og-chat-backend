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

@app.get("/debug")
def debug():
    try:
        tee_llm_values = [e.name for e in og.TEE_LLM]
        return {"TEE_LLM_values": tee_llm_values}
    except Exception as e:
        return {"error": str(e), "og_attrs": [x for x in dir(og) if not x.startswith("_")]}

@app.get("/")
def root():
    return {"status": "ok"}
```

Commit → wait for Railway to go green → open this in your browser:
```
https://og-chat-backend-production.up.railway.app/debug
