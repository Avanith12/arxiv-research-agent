import os

import gradio as gr
from dotenv import load_dotenv
from smolagents import CodeAgent, InferenceClientModel, OpenAIModel
from smolagents.memory import ActionStep

from tools import get_arxiv_paper, read_arxiv_pdf, search_arxiv

load_dotenv(".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")


def make_model():
    # OpenRouter keys usually start with sk-or-  Hugging Face keys start with hf_
    token = OPENROUTER_API_KEY or HF_TOKEN
    if not token:
        raise ValueError("Add OPENROUTER_API_KEY or HF_TOKEN to .env")

    if token.startswith("hf_"):
        return InferenceClientModel(
            model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
            token=token,
        )

    return OpenAIModel(
        model_id="openrouter/free",
        api_base="https://openrouter.ai/api/v1",
        api_key=token,
        client_kwargs={
            "default_headers": {
                "HTTP-Referer": "http://localhost:7860",
                "X-Title": "arXiv Research Agent",
            }
        },
    )

MAX_STEPS = 5
MAX_QUESTION_CHARS = 2000

INSTRUCTIONS = """
You are an arXiv research assistant.

You may ONLY help with academic papers and research.
If the user asks about weather, news, sports, or anything unrelated to papers,
refuse politely and say this app only searches academic papers.

Rules:
- Always use tools to look up papers. Never invent titles, authors, links, or citations.
- If a tool returns no results or TOOL FAILED, say so clearly.
- After finding papers, write a short summary of each abstract (3-5 sentences max).
- Include title, authors, year, and the arXiv URL for every paper.
- If the user asks for BibTeX, paste the bibtex field from the tool output.
- For a deeper look at ONE paper, call read_arxiv_pdf on its arxiv_id (first pages only).
- Keep answers in markdown. Do not use asterisks for decoration beyond markdown lists.
"""

model = make_model()

agent = CodeAgent(
    tools=[search_arxiv, get_arxiv_paper, read_arxiv_pdf],
    model=model,
    add_base_tools=False,
    max_steps=MAX_STEPS,
    instructions=INSTRUCTIONS,
)


def _message_text(message) -> str:
    if isinstance(message, dict):
        return str(message.get("content") or message.get("text") or "").strip()
    return str(message).strip()


def _history_is_empty(history) -> bool:
    return not history


def _format_trace() -> str:
    tool_names = []
    n_action = 0
    for step in agent.memory.steps:
        if not isinstance(step, ActionStep):
            continue
        n_action += 1
        if step.tool_calls:
            for call in step.tool_calls:
                tool_names.append(call.name)
    tools_used = ", ".join(tool_names) if tool_names else "none"
    return (
        f"\n\n---\n**Agent trace:** {n_action} ReAct steps "
        f"(max {MAX_STEPS}) | tools used: {tools_used}"
    )


def chat(message, history):
    user_text = _message_text(message)

    if not user_text:
        return "Please type a research question."
    if len(user_text) > MAX_QUESTION_CHARS:
        return (
            "Guardrail: please keep your question under "
            f"{MAX_QUESTION_CHARS} characters."
        )

    try:
        result = agent.run(user_text, reset=_history_is_empty(history))
    except Exception as exc:
        return f"The agent hit an error: {exc}\nI will not invent papers."

    return str(result) + _format_trace()


demo = gr.ChatInterface(
    fn=chat,
    title="arXiv Research Agent",
    description=(
        "Type your question in the box at the bottom. "
        "Ask for papers, summaries, BibTeX, or a short PDF read."
    ),
    textbox=gr.Textbox(
        placeholder="Type here, e.g. Find 3 papers on agentic AI and summarize them",
        lines=2,
        scale=7,
    ),
    examples=[
        "Find 3 papers on agentic AI and summarize them",
        "Search neural SDEs time series and give BibTeX",
        "Get paper 1706.03762, summarize it, and read the first PDF pages",
        "What is the weather in Boston?",
    ],
)

if __name__ == "__main__":
    demo.launch()
