import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import anthropic
from pydantic import BaseModel
from typing import Optional, List

load_dotenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise RuntimeError("ANTHROPIC_API_KEY manquante dans le fichier .env")

client = anthropic.Anthropic(api_key=api_key)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[str] = None


@app.get("/")
def serve_frontend():
    return FileResponse(BASE_DIR / "index.html")


@app.post("/api/chat")
def chat(req: ChatRequest):
    try:
        params = {
            "model": req.model,
            "max_tokens": req.max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in req.messages],
        }
        if req.system:
            params["system"] = req.system

        response = client.messages.create(**params)

        return {"content": [{"text": response.content[0].text}]}

    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=e.status_code, detail=str(e.message))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
