# llm_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama

app = FastAPI()
llm = Llama(model_path="mistral-7b-instruct-v0.1.Q2_K.gguf", n_ctx=2048, n_threads=6, n_gpu_layers=32)

class Prompt(BaseModel):
    prompt: str

@app.post("/generate")
def generate(p: Prompt):
    out = llm(p.prompt, max_tokens=256, stop=["</s>"])
    return {"response": out["choices"][0]["text"].strip()}