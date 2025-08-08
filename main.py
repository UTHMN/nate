# Fastapi
from fastapi import FastAPI
from pydantic import BaseModel

# llm
from llm import LLM

# Audio
import audio
import hashlib
import os
from fastapi import UploadFile, File, Form, HTTPException

app = FastAPI()
model = LLM()

# TODO: Return valid routes (not subroutes)
@app.get("/")
async def root():
    return {
        "message": {
            "/": "Displays this message",
            "/ask": "Query the AI",
            "/audio/*": "Multiple audio endpoints",
        }
    }

# Define a request model
class PromptRequest(BaseModel):
    prompt: str
    user: str # TODO: convert to tokens instead of username

@app.post("/ask")
async def ask(request: PromptRequest):
    return {"message": model.ask(request.prompt, request.user)}

@app.get("/audio")
async def audio_root():
    return {
        "message": {
            "/audio/transcribe": "Transcribes an mp3 file and returns the transcript",
            "/audio/enroll": "Enrolls a user for voice identification, also supports adding embeddings to a person",
            "/audio/transcribe_identify": "Transcribes an mp3 audio file and also sends the user identified"
        }
    }

# send audio file, auto converts to working formats, transcribes
def save_file_with_hash(file_bytes: bytes, ext: str) -> str:
    h = hashlib.sha256(file_bytes).hexdigest()
    filename = f"/tmp/{h}.{ext}"
    if not os.path.exists(filename):
        with open(filename, "wb") as f:
            f.write(file_bytes)
    return filename

def delete_file(path: str):
    try:
        os.remove(path)
    except Exception:
        pass  # silently ignore errors on delete

@app.post("/audio/transcribe")
async def audio_transcribe(file: UploadFile = File(...)):
    file_bytes = await file.read()
    ext = file.filename.split(".")[-1].lower()

    if ext not in ("mp3", "wav"):
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    file_path = save_file_with_hash(file_bytes, ext)
    try:
        # Assuming your audio.transcribe expects a file path
        transcript = audio.transcribe(file_path)  
        return {"transcript": transcript}
    finally:
        delete_file(file_path)

@app.post("/audio/enroll")
async def audio_enroll(
    file: UploadFile = File(...),
    user: str = Form(...)
):
    file_bytes = await file.read()
    ext = file.filename.split(".")[-1].lower()

    if ext not in ("mp3", "wav"):
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    file_path = save_file_with_hash(file_bytes, ext)
    try:
        # enroll_speaker expects a name and a file path
        audio.enroll_speaker(user, file_path)
        return {"message": f"User {user} enrolled successfully."}
    finally:
        delete_file(file_path)

@app.post("/audio/transcribe_identify")
async def audio_transcribe_identify(file: UploadFile = File(...)):
    file_bytes = await file.read()
    ext = file.filename.split(".")[-1].lower()

    if ext not in ("mp3", "wav"):
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    file_path = save_file_with_hash(file_bytes, ext)
    try:
        result = audio.classify_and_transcribe(file_path)
        return result
    finally:
        delete_file(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
