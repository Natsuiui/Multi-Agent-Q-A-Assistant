from llama_cpp import Llama

# Load Mistral 7B Instruct
llm = Llama.from_pretrained(
    repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
    filename="mistral-7b-instruct-v0.1.Q2_K.gguf",
    n_ctx=2048,
    n_threads=6,
    n_gpu_layers=32
)

# Top 3 retrieved chunks
context = """
1. Product: KSTech X10 Smartphone
- Display: 6.5" AMOLED
- Processor: Snapdragon 8 Gen 2
- RAM: 12GB
- Storage: 256GB
- Battery: 5000mAh
- OS: KS_OS 3.1

2. All KSTech products come with a 1-year limited warranty. The warranty covers manufacturing defects only and does not include accidental damage, water damage, or unauthorized repairs.

3. Q: What products does KSTech offer?
A: We offer smartphones, laptops, tablets, and accessories.

Q: Where do you ship?
A: We currently ship to all 28 Indian states.

Q: What is your return policy?
A: Returns are accepted within 30 days of purchase with original packaging.
"""

# User query
query = "What warranty comes with a smartphone?"

# Prompt template
prompt = f"""[INST]
Using the following context, answer the question clearly and concisely.

Context:
{context}

Question: {query}
[/INST]
"""


output = llm(prompt, max_tokens=256, stop=["</s>"])
print("📢 LLM Answer:\n")
print(output["choices"][0]["text"].strip())