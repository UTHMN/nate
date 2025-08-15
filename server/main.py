# Fastapi
from fastapi import FastAPI
from pydantic import BaseModel

# llm
from llm import LLM

# Audio
import audio as audio
import hashlib
import os
from fastapi import UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

# Tokens
from tokens import TokenManager

app = FastAPI()
model = LLM()
tokenManage = TokenManager()

# TODO: Return valid routes (not subroutes)
@app.get("/")
async def root():
    return {
        "message": {
            "/": "Displays this message",
            "/docs": "OpenAPI documentation",
            "/remove": "Remove a user from the database",
            "/messages/*": "Multiply query related endpoints",
            "/audio/*": "Multiple audio endpoints",
        }
    }

# Define a request model
class PromptRequest(BaseModel):
    prompt: str
    token: str # TODO: convert to tokens instead of username
    
class EnrollRequest(BaseModel):
    username: str
    
class RemoveRequest(BaseModel):
    token: str

# TODO: Remove voice embeddings.
@app.post("/remove")
async def remove(request: RemoveRequest):
    try:
        model.delete_user(request.token)
        return {"message": f"User {request.token} removed successfully."}
    except ValueError as e:
        return {"error": f"Token {request.token} is invalid: {e}"}

@app.get("/messages")
async def messages_root():
    return {
        "message": {
            "/messages/ask": "Query the AI",
            "/messages/enroll": "Enrolls a user for and returns a token, also supports adding embeddings to a person",
        }
    }

@app.post("/messages/ask")
async def ask(request: PromptRequest):
    try:
        return {"message": model.ask(request.prompt, request.token)}
    except ValueError as e:
        return {"error": f"Token {request.token} is invalid: {e}"}

@app.post("/messages/enroll")
async def enroll(request: EnrollRequest):
    try:
        token = model.enroll_user(request.username)
        return {"token": token}
    except ValueError as e:
        return {"error": f"User {request.username} is already enrolled: {e}"}

@app.get("/audio")
async def audio_root():
    return {
        "message": {
            "/audio/transcribe": "Transcribes an mp3 file and returns the transcript",
            "/audio/enroll": "Enrolls a user for voice identification, also supports adding embeddings to a person",
            "/audio/transcribe_identify": "Transcribes an mp3 audio file and also sends the user identified",
            "/audio/tts": "Converts text to speech using Google TTS"
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
    token: str = Form(...)
):
    file_bytes = await file.read()
    ext = file.filename.split(".")[-1].lower()

    if ext not in ("mp3", "wav"):
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    file_path = save_file_with_hash(file_bytes, ext)
    try:
        # enroll_speaker expects a name and a file path
        audio.enroll_speaker(token, file_path)
        return {"message": f"User {token} enrolled successfully."}
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

def delete_file(path: str):
    try:
        os.remove(path)
    except Exception as e:
        print(f"Error deleting file: {e}")

@app.post("/audio/tts")
async def audio_tts(text: str, background_tasks: BackgroundTasks):
    filepath = audio.tts(text)  # Generate the MP3 file
    background_tasks.add_task(delete_file, filepath)
    return FileResponse(filepath, media_type="audio/mpeg", filename="output.mp3")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
