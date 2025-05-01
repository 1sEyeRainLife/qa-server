import os
import uvicorn
from dotenv import load_dotenv
load_dotenv()
import asyncio
import json
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from agent import PDFQAAgent
from chat import chat
from _agent import agent as tool_agent


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
templates = Jinja2Templates(directory="static")
# auth middleware
qa_agent = PDFQAAgent(persist_db=True)

# 确保上传目录存在
UPLOAD_DIR = "uploads"
Path(UPLOAD_DIR).mkdir(exist_ok=True)
qa_history = []

class Question(BaseModel):
    text: str

class Knownledge(BaseModel):
    feedback_id: str
    user_id: str
    content: str
    rating: int = 0
    timestamp: str = ""

@app.post("/api/chat")
async def get_chat(question: Question):
    message = chat(question.text)
    qa_history.append({
        "question": question.text,
        "answer": message
    })
    return {"message": message}

@app.post("/api/tool/agent")
async def get_tool_agent(question: Question):
    return {"message": tool_agent.run(question.text)}

@app.post("/api/answers")
async def gen_answers(question: Question):

    answer = qa_agent.ask(question.text)
    return {"message": answer}

@app.post("/api/qa_knownledge1")
async def create_knownledge(knownledge: Knownledge):
    qa_agent.add_feedback(knownledge.model_dump())
    return {"ok": True}

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(str(file_path), "wb") as f:
        f.write(file.file.read())

    qa_agent.load_pdf(file_path)
    return {"ok": True, "file_path": file_path}

@app.get("/api/collections")
async def get_collections():
    return {"collections": qa_agent.list_collections()}

@app.post("/api/auth/token")
async def gen_token(username: str, password: str):
    return {"token": "<token>"}

@app.get("/api/auth/users")
async def list_users(page: int, limit: int):
    return {"page": page, "items": []}

@app.post("/api/auth/users")
async def create_users(data: dict):
    return {"id": "<uid>"}

@app.post("/api/documents")
async def upload_documents():
    return {"doc_id": "<>"}

@app.get("/api/analysis")
async def get_analysis():
    return {"data": {}}


# SSE 事件流生成器
async def event_generator(uid: str, qwen: str):
    while True:
        if qa_history:
            last_qa = qa_history.pop()
            answer = json.dumps(last_qa, ensure_ascii=False)
            # yield f"event: new_question\ndata: {last_qa['question']}\n\n"
            yield f"event: new_answer\ndata: {answer}\n\n"
        # else:
        #     current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #     yield f"event: time\ndata: {current_time}\n\n"

        await asyncio.sleep(1)

@app.get("/stream")
async def stream(uid: str, model: str):
    return StreamingResponse(
        event_generator(uid, model),
        media_type="text/event-stream"
    )

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    context = {
        "request": request,
        "title": "hello",
        "message": "h"
    }
    return templates.TemplateResponse("index.html", context)


if __name__ == "__main__":
    uvicorn.run("app:app", port=8001, reload=False)