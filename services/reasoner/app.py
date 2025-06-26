# services/reasoner/app.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from python_a2a.mcp import FastMCP, text_response
from python_a2a.mcp.transport import create_fastapi_app

# Use a small general-purpose model (e.g., GPT2)
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

mcp = FastMCP(name="reasoner")

@mcp.tool("solve-math")
async def solve_math(question: str) -> dict[str, str]:
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=80)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text_response(answer)

app = create_fastapi_app(mcp)
