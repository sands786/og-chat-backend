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

        print("RESULT TYPE:", type(result))
        print("RESULT ATTRS:", [a for a in dir(result) if not a.startswith('_')])
        print("RESULT DICT:", result.__dict__ if hasattr(result, '__dict__') else 'N/A')

        content = result.chat_output["content"]

        payment_hash = getattr(result, 'payment_hash', None)
        if not payment_hash:
            for field in ['transaction_hash', 'tx_hash', 'txHash', 'receipt_hash']:
                val = getattr(result, field, None)
                if val:
                    payment_hash = str(val)
                    break

        print(f"PAYMENT HASH: {payment_hash}")

        return {
            "content": content,
            "payment_hash": str(payment_hash) if payment_hash else "",
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
