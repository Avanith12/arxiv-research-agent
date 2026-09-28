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

A chat app that searches arXiv for research papers, summarizes them, and gives citations. Built with [smolagents](https://github.com/huggingface/smolagents) and [Gradio](https://www.gradio.app/).

## Features

- Search arXiv by topic, author, category (`cs.AI`, `cs.LG`, ...), or year
- Sort by newest or by relevance
- Short summary of each paper
- Structured summary from the PDF (Problem, Method, Results, Limitations)
- Compare two papers
- BibTeX citations
- Beginner mode for simpler explanations
- Download results as `.bib` or the chat as `.md`
- Follow-up questions in the same chat
- Agent trace showing ReAct loops and tools used

## Guardrails

- Max 8 agent steps per question
- Max 5 papers per search
- PDF reading limited to the first 3 pages
- Questions limited to 2,000 characters
- Instructed not to invent papers; tool errors are reported
- Off-topic questions are refused

## Tools

| Tool | Purpose |
|---|---|
| `search_arxiv` | Search with topic, author, category, and year filters |
| `get_arxiv_paper` | Get one paper by arXiv ID |
| `compare_papers` | Get two papers for comparison |
| `read_arxiv_pdf` | Read text from the first pages of a PDF |

## Run locally

```bash
git clone https://github.com/Avanith12/arxiv-research-agent.git
cd arxiv-research-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:

```
OPENROUTER_API_KEY=your_key_here
```

Start the app and open http://127.0.0.1:7860:

```bash
python app.py
```

## Deploy on Hugging Face Spaces

Add `OPENROUTER_API_KEY` as a secret in **Settings → Variables and secrets**.

## Example questions

- Find 3 papers on agentic AI and summarize them
- Latest cs.LG papers on world models since 2025
- Find 2 recent papers by Yann LeCun
- Compare 1706.03762 and 1810.04805
- Give me a structured summary of 1706.03762

## Credits

Built for the AI Industry Lab class with Professor Wei Ding. Based on Inal's arXiv agent reference implementation, extended with real agent calls, filtered search, paper comparison, structured PDF summaries, beginner mode, BibTeX and chat downloads, guardrails, and a chat interface.
