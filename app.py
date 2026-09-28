import os
import tempfile
from datetime import datetime

import gradio as gr
from dotenv import load_dotenv
from smolagents import CodeAgent, InferenceClientModel, OpenAIModel
from smolagents.memory import ActionStep

from tools import (
    SAVED_PAPERS,
    clear_saved_papers,
    compare_papers,
    get_arxiv_paper,
    read_arxiv_pdf,
    search_arxiv,
)

load_dotenv(".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")


def make_model():
    # OpenRouter keys usually start with sk-or-  Hugging Face keys start with hf_
    token = OPENROUTER_API_KEY or HF_TOKEN
    if not token:
        raise ValueError(
            "Add OPENROUTER_API_KEY to .env (locally) or as a Space secret (on Hugging Face)"
        )

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


MAX_STEPS = 8
MAX_QUESTION_CHARS = 2000

INSTRUCTIONS = """
You are an arXiv research assistant.

You may ONLY help with academic papers and research.
If the user asks about weather, news, sports, or anything unrelated to papers,
refuse politely and say this app only searches academic papers.

Tools:
- search_arxiv: search by topic, author, category (cs.AI, cs.LG, cs.CL, stat.ML...),
  since_year, and newest_first. Use newest_first=True for "latest" or "recent" papers.
  Use the author argument for "papers by <name>".
- get_arxiv_paper: one paper by arXiv id.
- compare_papers: two papers side by side.
- read_arxiv_pdf: text from the first pages of one paper's PDF.

Rules:
- Always use tools to look up papers. Never invent titles, authors, links, or citations.
- If a tool returns an error or TOOL FAILED, say so clearly.
- For every paper, include title, authors, year, and the arXiv URL.
- After finding papers, write a short summary of each abstract (3-5 sentences max).
- If the user asks for BibTeX, paste the bibtex field from the tool output in a code block.

Comparing papers:
- Call compare_papers. Your final answer MUST contain a markdown table with one
  column per paper and these rows: Problem, Approach, Key results, Year, Authors.
- After the table, add one sentence on which paper to read first and why.

Detailed / structured summary ("deep dive", "explain the paper", "structured summary"):
- You MUST call read_arxiv_pdf with max_pages=3. The abstract alone is not enough.
- Answer with these headings: Problem, Method, Results, Limitations.
- If a section is not in the text you read, write "Not covered in the first pages."

Keep answers in markdown.

How to reply:
- Every reply must be a code block that ends by calling final_answer(...).
- This includes refusals and clarifying questions. Never reply with plain text only.
- Always pass the answer in triple quotes so multi-line text is valid Python:
  <code>
  final_answer(\"\"\"Here are the papers:

  1. Title - Authors (Year)
     URL
  \"\"\")
  </code>
"""

BEGINNER_PREFIX = (
    "Explain simply for a beginner: avoid jargon, use short sentences, "
    "define any technical term you use, and give one everyday analogy per paper.\n\n"
)

model = make_model()

agent = CodeAgent(
    tools=[search_arxiv, get_arxiv_paper, compare_papers, read_arxiv_pdf],
    model=model,
    add_base_tools=False,
    max_steps=MAX_STEPS,
    instructions=INSTRUCTIONS,
)


def _message_text(message) -> str:
    if isinstance(message, dict):
        content = message.get("content") or message.get("text") or ""
    else:
        content = message
    if isinstance(content, list):
        content = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content).strip()


ARXIV_TOOL_NAMES = [
    tool.name for tool in (search_arxiv, get_arxiv_paper, compare_papers, read_arxiv_pdf)
]


def _format_trace() -> str:
    # CodeAgent runs every action as python_interpreter, so read the code to see
    # which arXiv tools it actually called.
    loops = 0
    tools_used = []
    for step in agent.memory.steps:
        if not isinstance(step, ActionStep) or not step.code_action:
            continue
        loops += 1
        for name in ARXIV_TOOL_NAMES:
            if f"{name}(" in step.code_action and name not in tools_used:
                tools_used.append(name)
    return (
        f"\n\n---\n**Agent trace:** {loops} ReAct loops (step limit {MAX_STEPS}) | "
        f"arXiv tools used: {', '.join(tools_used) or 'none'}"
    )


def _clean_answer(result) -> str:
    # When the model keeps failing to write valid code, smolagents gives up at the step
    # limit and returns the model's raw text, e.g. final_answer("...") Calling tools: [...]
    text = str(result).strip()
    text = text.split("Calling tools:")[0].strip()
    if text.startswith("final_answer("):
        text = text[len("final_answer("):]
        if text.endswith(")"):
            text = text[:-1]
        text = text.strip().strip('"').strip("'").strip()
    return text.replace("\\n", "\n").replace('\\"', '"')


def chat(message, history, simple_mode):
    user_text = _message_text(message)

    if not user_text:
        return "Please type a research question."
    if len(user_text) > MAX_QUESTION_CHARS:
        return (
            "Guardrail: please keep your question under "
            f"{MAX_QUESTION_CHARS} characters."
        )

    new_conversation = not history
    if new_conversation:
        clear_saved_papers()

    task = BEGINNER_PREFIX + user_text if simple_mode else user_text

    try:
        result = agent.run(task, reset=new_conversation)
    except Exception as exc:
        return f"The agent hit an error: {exc}\nI will not invent papers."

    return _clean_answer(result) + _format_trace()


def _write_download(filename: str, text: str) -> str:
    path = os.path.join(tempfile.mkdtemp(), filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def export_bibtex():
    if not SAVED_PAPERS:
        gr.Warning("No papers yet. Ask the agent to find some papers first.")
        return None
    text = "\n\n".join(paper["bibtex"] for paper in SAVED_PAPERS.values()) + "\n"
    return _write_download("arxiv_papers.bib", text)


def export_chat(history):
    if not history:
        gr.Warning("The chat is empty. Ask a question first.")
        return None
    lines = [f"# arXiv Research Agent chat ({datetime.now():%Y-%m-%d %H:%M})\n"]
    for message in history:
        role = message.get("role", "") if isinstance(message, dict) else ""
        speaker = "You" if role == "user" else "Agent"
        lines.append(f"## {speaker}\n\n{_message_text(message)}\n")
    return _write_download("arxiv_chat.md", "\n".join(lines))


EXAMPLES = [
    "Find 3 papers on agentic AI and summarize them",
    "Latest cs.LG papers on world models since 2025",
    "Find 2 recent papers by Yann LeCun",
    "Compare 1706.03762 and 1810.04805",
    "Give me a structured summary of 1706.03762",
    "Search neural SDEs time series and give BibTeX",
]


def add_user_message(message, history):
    text = _message_text(message)
    if not text:
        return message, history
    return "", history + [{"role": "user", "content": text}]


def add_agent_reply(history, simple_mode):
    if not history or history[-1].get("role") != "user":
        return history
    question = _message_text(history[-1])
    answer = chat(question, history[:-1], simple_mode)
    return history + [{"role": "assistant", "content": answer}]


with gr.Blocks(title="arXiv Research Agent") as demo:
    gr.Markdown(
        "# arXiv Research Agent\n"
        "Search by topic, author, field, or year, compare two papers, get a structured "
        "summary from the PDF, or ask for BibTeX. Type below and press **Send**."
    )

    chatbot = gr.Chatbot(label="Chat", height=480)

    with gr.Row():
        question_box = gr.Textbox(
            placeholder="Type here, e.g. Find 3 papers on agentic AI and summarize them",
            show_label=False,
            lines=2,
            scale=8,
        )
        send_button = gr.Button("Send", variant="primary", scale=1)

    with gr.Row():
        simple_mode = gr.Checkbox(label="Explain simply (beginner mode)", value=False)
        clear_button = gr.ClearButton([chatbot, question_box], value="New chat")

    gr.Examples(examples=EXAMPLES, inputs=question_box, label="Try an example")

    gr.Markdown("### Downloads")
    with gr.Row():
        bib_button = gr.Button("Download BibTeX (.bib)")
        chat_button = gr.Button("Download chat (.md)")
    download_file = gr.File(label="Your file", interactive=False)

    for trigger in (question_box.submit, send_button.click):
        trigger(
            add_user_message, [question_box, chatbot], [question_box, chatbot]
        ).then(add_agent_reply, [chatbot, simple_mode], chatbot)

    bib_button.click(export_bibtex, outputs=download_file)
    chat_button.click(export_chat, inputs=chatbot, outputs=download_file)


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Monochrome())
