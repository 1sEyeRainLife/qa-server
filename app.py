import os
import uvicorn
from dotenv import load_dotenv

load_dotenv()
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import PDFQAAgent


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
# auth middleware
qa_agent = PDFQAAgent(persist_db=True)

# 确保上传目录存在
UPLOAD_DIR = "uploads"
Path(UPLOAD_DIR).mkdir(exist_ok=True)


class Question(BaseModel):
    text: str

@app.post("/api/answers")
async def gen_answers(question: Question):

    answer = qa_agent.ask(question.text)
    return {"message": answer}

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


if __name__ == "__main__":
    uvicorn.run("app:app", reload=True)