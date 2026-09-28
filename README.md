---
title: arXiv Research Agent
emoji: ⚡
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 6.28.0
python_version: '3.12'
app_file: app.py
pinned: false
short_description: AI agent that searches and summarizes arXiv papers
---

# arXiv Research Agent

A chat app that finds real research papers on arXiv, summarizes them, and gives you citations. Built with [smolagents](https://github.com/huggingface/smolagents) and [Gradio](https://www.gradio.app/).

You ask a question in plain English. The agent decides which tools to call, looks up real papers, and answers. It never makes up papers.

## Features

- **Live paper search** — searches arXiv for real papers (up to 5 at a time)
- **Summaries** — a short summary of each paper's abstract
- **Look up one paper** — fetch a paper directly by its arXiv ID (for example `1706.03762`)
- **PDF reading** — downloads a paper's PDF and reads the first pages
- **BibTeX** — ready-to-paste citations for every paper
- **Chat with follow-ups** — ask "give me BibTeX for the second one" without repeating yourself
- **Agent trace** — each answer shows how many ReAct steps it took and which tools it used

## Guardrails

| Guardrail | What it prevents |
|---|---|
| Max 5 agent steps | The agent looping forever and burning API quota |
| Max 5 papers per search | Huge, slow responses |
| PDF capped at 3 pages / 8,000 characters | Overloading the model with a full paper |
| Questions capped at 2,000 characters | Oversized inputs |
| Never invent papers | If a search fails, the agent says so instead of making up results |
| Topic filter | Off-topic questions (weather, sports, etc.) are politely refused |

## How it works

The agent follows a **ReAct loop**: it reasons about the question, calls a tool, reads the result, and repeats until it can answer.

It has three tools, defined in `tools.py`:

| Tool | What it does |
|---|---|
| `search_arxiv` | Searches arXiv by keywords |
| `get_arxiv_paper` | Gets one paper by its arXiv ID |
| `read_arxiv_pdf` | Downloads a PDF and extracts text from the first pages |

The language model comes from [OpenRouter](https://openrouter.ai) (free tier). A Hugging Face token also works.

## Run it locally

```bash
git clone https://github.com/Avanith12/arxiv-research-agent.git
cd arxiv-research-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your key:

```
OPENROUTER_API_KEY=your_openrouter_key_here
```

Then start the app:

```bash
python app.py
```

Open http://127.0.0.1:7860 and type your question in the box at the bottom.

`.env` is in `.gitignore`, so your key is never committed.

## Deploy on Hugging Face Spaces

1. Push this repo to a Gradio Space.
2. In the Space, go to **Settings → Variables and secrets → New secret**.
3. Name it `OPENROUTER_API_KEY` and paste your key as the value.

## Example questions

- Find 3 papers on agentic AI and summarize them
- Search neural SDEs time series and give BibTeX
- Get paper 1706.03762, summarize it, and read the first PDF pages
- What is the weather in Boston? *(the agent refuses — it only handles papers)*

## Project structure

```
app.py            Gradio chat UI, agent setup, guardrails
tools.py          arXiv search, paper lookup, PDF reader
requirements.txt  Python dependencies
```

## Credits

Built for the AI Industry Lab class with Professor Wei Ding. Based on Inal's arXiv agent reference implementation, extended with real agent calls, PDF reading, summaries, BibTeX, guardrails, and a chat interface.
