import os
import uuid
import shutil
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from indexing_pipeline import index_document, query_document_filtered

app = FastAPI(title="Document Indexing API")

UPLOAD_DIR = "test-files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Track indexing status in memory — in production this would be Redis or a DB
document_status: dict[str, dict] = {}


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Accept a PDF upload, save it, and index it in the background
    so the response returns instantly instead of waiting for embedding.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    document_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{document_id}.pdf")

    # Save the uploaded file to disk
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    document_status[document_id] = {"status": "processing", "file_path": file_path}

    # Run indexing in the background — response doesn't wait for this
    background_tasks.add_task(run_indexing, document_id, file_path)

    return {
        "document_id": document_id,
        "status": "processing",
        "message": "Document uploaded, indexing started in background"
    }


def run_indexing(document_id: str, file_path: str):
    """Background task — runs after the response has already been sent."""
    try:
        result = index_document(file_path)
        document_status[document_id] = {
            "status": "indexed",
            "file_path": file_path,
            "chunks_created": result["chunks_created"]
        }
    except Exception as e:
        document_status[document_id] = {
            "status": "failed",
            "file_path": file_path,
            "error": str(e)
        }


@app.get("/documents/{document_id}/status")
def get_status(document_id: str):
    if document_id not in document_status:
        raise HTTPException(status_code=404, detail="Document not found")
    return document_status[document_id]


@app.post("/documents/{document_id}/query")
def query_indexed_document(document_id: str, question: str):
    if document_id not in document_status:
        raise HTTPException(status_code=404, detail="Document not found")

    status = document_status[document_id]
    if status["status"] != "indexed":
        raise HTTPException(status_code=400, detail=f"Document is not ready yet, status: {status['status']}")

    results = query_document_filtered(question, source_file=status["file_path"])
    return {"question": question, "results": results}