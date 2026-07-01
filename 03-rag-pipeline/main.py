import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from retrieval import index_resume
from rag_chain import ask_rag

load_dotenv()

app = FastAPI(title="RAG Pipeline API — Day 3")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

document_status: dict[str, dict] = {}


class QueryRequest(BaseModel):
    question: str
    document_id: str | None = None


@app.get("/")
def health_check():
    return {"status": "ok", "service": "RAG Pipeline API"}


@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files supported")

    document_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{document_id}.pdf")

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    document_status[document_id] = {
        "status": "processing",
        "file_path": file_path
    }

    background_tasks.add_task(run_indexing, document_id, file_path)

    return {
        "document_id": document_id,
        "status": "processing",
        "message": "Upload received, indexing in background"
    }


def run_indexing(document_id: str, file_path: str):
    try:
        result = index_resume(file_path, collection_name=document_id)
        document_status[document_id] = {
            "status": "indexed",
            "file_path": file_path,
            "chunks_created": result["chunks_indexed"]
        }
    except Exception as e:
        document_status[document_id] = {
            "status": "failed",
            "error": str(e)
        }


@app.get("/documents/{document_id}/status")
def get_status(document_id: str):
    if document_id not in document_status:
        raise HTTPException(status_code=404, detail="Document not found")
    return document_status[document_id]


@app.post("/query")
def query(request: QueryRequest):
    """
    The main RAG endpoint — retrieves relevant chunks,
    calls Gemini, returns answer + sources with citations.
    """
    collection_name = request.document_id or "resume"

    if request.document_id:
        if request.document_id not in document_status:
            raise HTTPException(status_code=404, detail="Document not found")
        if document_status[request.document_id]["status"] != "indexed":
            raise HTTPException(status_code=400, detail="Document not ready yet")

    result = ask_rag(request.question, collection_name=collection_name)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)