from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from backend import process_document, hybrid_retriever, generate_answer

app = FastAPI()

class Query(BaseModel):
    question: str
    search_type: str = "semantic"  # or "keyword"

@app.post("/upload/")
async def upload_document(
    file: UploadFile = File(None),
    url: str = Form(None)  # Accept URL as form data
):
    try:
        result = process_document(file, url)
        return {"status": "success", "documents_processed": result}
    except Exception as e:
        raise HTTPException(500, f"Processing error: {str(e)}")

@app.post("/query/")
async def answer_query(query: Query):
    def generate_stream():
        try:
            context_docs = hybrid_retriever(query.question, query.search_type)
            for chunk in generate_answer(query.question, context_docs):
                yield chunk
        except Exception as e:
            yield f"Error: {str(e)}".encode("utf-8")

    return StreamingResponse(generate_stream(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)