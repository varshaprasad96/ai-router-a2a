# services/summarizer/app.py
from fastapi import FastAPI
from transformers import pipeline
from python_a2a.mcp import FastMCP, text_response
from python_a2a.mcp.transport import create_fastapi_app

summarizer_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

# Create the FastMCP instance with a name
mcp = FastMCP(name="summarizer")

@mcp.tool("summarize")
async def summarize(text: str) -> dict[str, str]:
    summary = summarizer_pipeline(text, max_length=120, min_length=40, do_sample=False)
    return text_response(summary[0]["summary_text"])

# Create the FastAPI app from MCP
app = create_fastapi_app(mcp)
