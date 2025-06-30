# AI Router A2A

An intelligent routing system that uses semantic similarity to automatically route user requests to the most appropriate AI service. Built with FastAPI, Python A2A, and sentence transformers.

## 🏗️ Architecture

The project consists of three main components:

- **Orchestrator**: Intelligent router that uses cosine similarity to select the best service
- **Summarizer Service**: Text summarization using BART model
- **Reasoner Service**: Math problem solving using Phi-2 model

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Client    │───▶│ Orchestrator │───▶│ Summarizer  │
│             │    │              │    │ Service     │
└─────────────┘    └──────────────┘    └─────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │ Reasoner    │
                   │ Service     │
                   └─────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Virtual environments for each service
- Required Python packages (see installation below)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-router-a2a
   ```

2. **Set up Summarizer Service**
   ```bash
   cd services/summarizer
   source .venv/bin/activate
   pip install fastapi transformers torch python-a2a uvicorn sentence-transformers
   ```

3. **Set up Reasoner Service**
   ```bash
   cd services/reasoner
   source .venv/bin/activate
   pip install fastapi transformers torch python-a2a uvicorn sentence-transformers
   ```

4. **Set up Orchestrator**
   ```bash
   cd orchestrator
   source .venv/bin/activate
   pip install fastapi python-a2a uvicorn sentence-transformers httpx
   ```

### Running the Services

1. **Start Summarizer Service** (Port 8001)
   ```bash
   cd services/summarizer
   source .venv/bin/activate
   uvicorn app:app --host 0.0.0.0 --port 8001 --reload
   ```

2. **Start Reasoner Service** (Port 9000)
   ```bash
   cd services/reasoner
   source .venv/bin/activate
   uvicorn app:app --host 0.0.0.0 --port 9000 --reload
   ```

3. **Start Orchestrator** (Port 8002)
   ```bash
   cd orchestrator
   source .venv/bin/activate
   uvicorn app:app --host 0.0.0.0 --port 8002 --reload
   ```

## 📡 API Usage

### Demo

Watch a live demo of the orchestrator in action:

[![asciicast](https://asciinema.org/a/725259.svg)](https://asciinema.org/a/725259)

Alternatively, you can play the demo locally:
```bash
# Install asciinema if you haven't already
pip install asciinema

# Play the demo
asciinema play orchestrator-demo.cast
```

### Orchestrator Endpoint

**POST** `/plan`

Send a prompt to the orchestrator, and it will automatically route to the best service.

```bash
curl -X POST "http://localhost:8002/plan" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Summarize this long article about AI"}'
```

**Example Responses:**

For summarization requests:
```json
{
  "content": "This is a concise summary of the article..."
}
```

For math problems:
```json
{
  "content": "The answer is 42..."
}
```

### Direct Service Endpoints

**Summarizer Service:**
```bash
curl -X POST "http://localhost:8001/tools/summarize" \
     -H "Content-Type: application/json" \
     -d '{"text": "Your long text to summarize"}'
```

**Reasoner Service:**
```bash
curl -X POST "http://localhost:9000/tools/solve-math" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is 2 + 2?"}'
```

## 🧠 How It Works

### Intelligent Routing

The orchestrator uses **cosine similarity** with sentence embeddings to determine the most appropriate service:

1. **Embedding Generation**: User prompt is converted to embeddings using `all-MiniLM-L6-v2`
2. **Similarity Calculation**: Cosine similarity is computed between prompt and service descriptions
3. **Service Selection**: The service with highest similarity score is selected
4. **Request Routing**: HTTP request is sent to the selected service

### Service Descriptions

- **Summarizer**: "Summarizes long articles or text passages."
- **Reasoner**: "Solves math problems and logical reasoning questions."

### Example Routing

| User Prompt | Selected Service | Similarity Score |
|-------------|------------------|------------------|
| "Summarize this article" | Summarizer | 0.85 |
| "What is 15 + 27?" | Reasoner | 0.78 |
| "Give me a brief overview" | Summarizer | 0.72 |

## 🛠️ Technical Details

### Models Used

- **Summarizer**: `facebook/bart-large-cnn` (Hugging Face)
- **Reasoner**: `microsoft/phi-2` (Hugging Face)
- **Embeddings**: `all-MiniLM-L6-v2` (Sentence Transformers)

### Technologies

- **FastAPI**: Web framework for all services
- **Python A2A**: MCP (Model Context Protocol) implementation
- **Sentence Transformers**: Semantic similarity calculations
- **Transformers**: Hugging Face model loading and inference
- **PyTorch**: Deep learning backend

### Architecture Benefits

- **Scalable**: Easy to add new services
- **Intelligent**: Automatic service selection based on content
- **Modular**: Each service is independent
- **Fast**: Lightweight embedding-based routing

## 🔧 Configuration

### Adding New Services

1. Create a new service directory in `services/`
2. Implement the service using FastMCP
3. Add service description to orchestrator's `TOOLS` dictionary
4. Update the routing logic if needed

### Customizing Models

Each service can be easily modified to use different models by changing the model name in the respective `app.py` files.

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all packages are installed in the correct virtual environment
2. **Timeout Errors**: The reasoner service may take time to load the model. Increase timeout in orchestrator if needed
3. **Port Conflicts**: Make sure ports 8001, 8002, and 9000 are available
