import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

app = FastAPI(title="RepoMind Backend")

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"status": "RepoMind backend is running!"}

@app.post("/generate")
def generate_ai_response(data: PromptRequest):
    if not api_key:
        return {"error": "GROQ_API_KEY is missing from .env file"}

    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",  # Updated to currently active Groq model
        messages=[{"role": "user", "content": data.prompt}]
    )
    
    return {"response": completion.choices[0].message.content}