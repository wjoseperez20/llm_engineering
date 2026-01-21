# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an 8-week LLM Engineering course repository created by Edward Donner. The course teaches students to build AI applications, progressing from basic LLM API calls to an autonomous multi-agent system. The repository contains educational materials, Jupyter notebooks, Python scripts, and community contributions.

**Key Learning Path:**
- Week 1: Basic LLM API calls, web scraping, summarization
- Week 2: Prompt engineering, function calling, structured outputs
- Week 3: HuggingFace pipelines, tokenizers, transformers (GPU-based via Google Colab)
- Week 4: Building AI applications with gradio frontends
- Week 5: RAG (Retrieval Augmented Generation) with vector databases
- Week 6: Fine-tuning LLMs on custom datasets
- Week 7: Model training and deployment on Modal
- Week 8: Multi-agent autonomous systems

## Environment Setup

This project uses `uv` for Python package management (not Anaconda). The environment is defined in:
- `pyproject.toml` - Dependencies and project configuration
- `uv.lock` - Locked dependency versions
- `.env` - API keys (OPENAI_API_KEY, GOOGLE_API_KEY)
- `.python-version` - Python version specification

**Python Version:** 3.11+

### Common Commands

**Initial setup:**
```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

**Running notebooks:**
```bash
# Start Jupyter (notebooks are typically run in Cursor with Jupyter extension)
jupyter lab
```

**Running Python scripts:**
```bash
# Week 1 example
python week1/scraper.py

# Week 8 autonomous agent application
python week8/price_is_right.py
```

**Week 3-7 Google Colab:**
Week 3 onwards includes GPU-intensive tasks that run on Google Colab (links in notebooks and README). Local execution is optional for these weeks.

## Project Structure

```
llm_engineering/
├── week1-8/              # Weekly course content
│   ├── day*.ipynb        # Daily Jupyter notebooks with exercises
│   ├── *.py              # Python utility scripts
│   └── community-contributions/  # Student submissions
├── guides/               # Technical guides (Python, Git, APIs, etc.)
├── setup/               # Platform-specific setup instructions
├── assets/              # Images and resources for notebooks
├── extras/              # Supplementary materials
└── .env                 # API keys (not committed to git)
```

## Architecture Patterns

### Week 1-2: Basic LLM Integration
- Uses OpenAI client directly
- Message format: `[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]`
- Common pattern: system prompt + user prompt with dynamic content
- Web scraping with BeautifulSoup (`scraper.py`)

### Week 5: RAG Implementation
- ChromaDB for vector storage
- LangChain for orchestration (updated to v1.0+ as of November 2025)
- Pattern: retrieve relevant context → augment prompt → query LLM
- Gradio for web UI (`app.py`, `gradio.ChatInterface`)

### Week 6: Fine-tuning
- JSONL format for training data
- OpenAI fine-tuning API
- Custom pricer models for product pricing

### Week 7: Model Deployment
- Modal for cloud deployment
- Custom inference endpoints
- Integration with frontends

### Week 8: Multi-Agent Framework
**File:** `week8/deal_agent_framework.py`

The autonomous agent system follows a hierarchical architecture:

```
DealAgentFramework (Orchestrator)
├── PlanningAgent (Coordinator)
│   ├── ScannerAgent (Web scraping for deals)
│   ├── EnsembleAgent (Price estimation)
│   │   ├── NeuralNetworkAgent (Fine-tuned model on Modal)
│   │   └── FrontierAgent (RAG with GPT-4)
│   ├── EvaluatorAgent (Deal quality assessment)
│   └── MessagingAgent (Push notifications)
```

**Key patterns:**
- Agent base class inheritance (`agents/agent.py`)
- Stateful memory with JSON persistence (`memory.json`)
- ChromaDB vector store for RAG (`products_vectorstore/`)
- Gradio UI with real-time logging and 3D visualization
- 5-minute polling timer for autonomous operation

**Agent communication:**
- Planning agent coordinates sub-agents
- Results flow up through agent hierarchy
- Opportunities stored in shared memory
- Logging with structured format for UI display

## API Keys and Services

Required API keys in `.env`:
- `OPENAI_API_KEY` - For GPT models (gpt-4.1-mini, gpt-4.1-nano, gpt-5-nano)
- `GOOGLE_API_KEY` - For Gemini models (optional)

**Cost management:** The course emphasizes minimal API costs (a few cents per exercise). Prefer smaller models like `gpt-4.1-nano` for development.

**Ollama alternative:** Free local models can be used instead of paid APIs. See `guides/09_ai_apis_and_ollama.ipynb` for instructions.

## Testing and Development

This is an educational repository with minimal formal testing:
- Primary development in Jupyter notebooks (`.ipynb` files)
- Experimentation encouraged via notebook cells
- Community contributions in `week*/community-contributions/` folders
- No standard test suite; validation through notebook execution

**Development workflow:**
1. Watch lecture video
2. Execute corresponding notebook cell-by-cell
3. Experiment with variations
4. Optionally submit solutions via Pull Request

## Common Patterns and Utilities

**LLM API calls:**
```python
from openai import OpenAI
openai = OpenAI()
response = openai.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
)
result = response.choices[0].message.content
```

**Web scraping:**
```python
from scraper import fetch_website_contents
content = fetch_website_contents("https://example.com")
```

**Gradio UI:**
```python
import gradio as gr
view = gr.ChatInterface(chat_function, type="messages").launch(inbrowser=True)
```

**Vector DB (Week 5+):**
```python
import chromadb
client = chromadb.PersistentClient(path="vectorstore")
collection = client.get_or_create_collection("docs")
```

## Important Notes

1. **Course Updates:** Repository updated regularly with improved explanations and new models. Code may differ slightly from videos.

2. **Platform:** Designed for Cursor IDE with Jupyter extension. Kernel selection required for each notebook.

3. **Community Contributions:** Students encouraged to share solutions. Place contributions in appropriate `community-contributions/` folder.

4. **Google Colab:** Week 3+ includes GPU-intensive work. Colab links provided in notebooks and README.

5. **Git Branches:**
   - `main` - New course version (uv-based, October 2025+)
   - `original` - Original course version (Anaconda-based)

6. **Modal Deployment (Week 7):** Requires Modal account and separate setup. Used for deploying fine-tuned models.

7. **ChromaDB:** Week 5+ uses persistent vector storage. Database stored in local directories (e.g., `week8/products_vectorstore/`).

## Troubleshooting

Refer to `setup/troubleshooting.ipynb` for common issues:
- API key configuration
- SSL/certificate errors (corporate networks)
- Import errors (missing dependencies)
- Jupyter kernel selection
- Path and permission issues

Contact: ed@edwarddonner.com or via LinkedIn for support.
