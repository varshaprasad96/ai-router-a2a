from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import httpx
import json

# Load local embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Define tool endpoints and descriptions
TOOLS = {
    "summarize": {
        "url": "http://localhost:8001/tools/summarize",
        "description": "Summarizes long articles or text passages."
    },
    "solve-math": {
        "url": "http://localhost:9000/tools/solve-math",
        "description": "Solves math problems and logical reasoning questions."
    },
}

class PromptInput(BaseModel):
    prompt: str

app = FastAPI()

async def choose_tool(prompt: str) -> str:
    # Embed user prompt
    prompt_emb = embedder.encode(prompt, convert_to_tensor=True)

    tool_embs = []
    tool_names = []
    for name, data in TOOLS.items():
        tool_names.append(name)
        emb = embedder.encode(data["description"], convert_to_tensor=True)
        tool_embs.append(emb)

    import torch
    tool_embs_tensor = torch.stack(tool_embs)  # stack into one tensor

    cos_scores = util.cos_sim(prompt_emb, tool_embs_tensor)[0]

    # Select tool with highest similarity
    best_idx = cos_scores.argmax().item()
    chosen_tool = tool_names[best_idx]
    print(f"[Planner] Selected tool: {chosen_tool} with score: {cos_scores[best_idx]:.4f}")
    return chosen_tool

@app.post("/plan")
async def plan_route(input: PromptInput):
    tool_name = await choose_tool(input.prompt)
    if tool_name not in TOOLS:
        return {"error": f"Unknown tool selected: {tool_name}"}

    # Choose input key based on tool
    input_key = "text" if tool_name == "summarize" else "question"
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:  # 60 second timeout
            response = await client.post(
                TOOLS[tool_name]["url"],
                json={input_key: input.prompt}
            )
            result = response.json()
            return result
    except httpx.TimeoutException:
        return {"error": f"Timeout calling {tool_name} service"}
    except httpx.RequestError as e:
        return {"error": f"Request error calling {tool_name} service: {str(e)}"}
    except Exception as e:
        return {"error": f"Error calling {tool_name} service: {str(e)}"}
