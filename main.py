# Fastapi
from fastapi import FastAPI
from pydantic import BaseModel

# llm
from llm import LLM

app = FastAPI()
model = LLM()

# TODO: Return valid routes (not subroutes)
@app.get("/")
async def root():
    return {"message": "Hello World"}

# Define a request model
class PromptRequest(BaseModel):
    prompt: str
    user: str # TODO: convert to tokens instead of username

@app.post("/ask")
async def ask(request: PromptRequest):
    return {"message": model.ask(request.prompt, request.user)}

# TODO: Return valid routes (not subroutes)
# send audio file, auto converts to working formats, transcribes and gets user from voice
# @app.post("/audio/transcribe")
# enrolls user to audio profiles 
# @app.post("/audio/enroll")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
