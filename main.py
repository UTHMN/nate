# Fastapi
from fastapi import FastAPI
from pydantic import BaseModel

# llm
from llm import LLM

app = FastAPI()
model = LLM()

@app.get("/")
async def root():
    return {"message": "Hello World"}

# Define a request model
class PromptRequest(BaseModel):
    prompt: str

@app.post("/ask")
async def ask(request: PromptRequest):
    return {"message": model.ask(request.prompt)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
