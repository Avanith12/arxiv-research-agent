import io
import urllib.request

import arxiv
from pypdf import PdfReader
from smolagents import tool


def _clean_arxiv_id(arxiv_id: str) -> str:
    value = (arxiv_id or "").strip()
    value = value.replace("https://arxiv.org/abs/", "")
    value = value.replace("http://arxiv.org/abs/", "")
    value = value.replace("https://arxiv.org/pdf/", "")
    value = value.replace("arxiv:", "")
    value = value.replace(".pdf", "")
    return value.strip()


def _paper_to_dict(paper) -> dict:
    arxiv_id = paper.get_short_id()
    authors = ", ".join(author.name for author in paper.authors)
    bibtex_authors = " and ".join(author.name for author in paper.authors)
    bibtex = (
        f"@article{{{arxiv_id},\n"
        f"  title = {{{paper.title}}},\n"
        f"  author = {{{bibtex_authors}}},\n"
        f"  year = {{{paper.published.year}}},\n"
        f"  eprint = {{{arxiv_id}}},\n"
        f"  archivePrefix = {{arXiv}},\n"
        f"  url = {{{paper.entry_id}}}\n"
        f"}}"
    )
    return {
        "arxiv_id": arxiv_id,
        "title": paper.title,
        "authors": authors,
        "year": paper.published.year,
        "url": paper.entry_id,
        "pdf_url": paper.pdf_url,
        "abstract": paper.summary,
        "bibtex": bibtex,
    }


@tool
def search_arxiv(query: str, max_results: int = 3) -> list:
    """
    Search arXiv for academic papers.

    Args:
        query: Search terms (e.g. 'agentic AI safety').
        max_results: How many papers to return. Capped at 5.

    Returns:
        A list of paper dicts with title, authors, year, url, pdf_url, abstract, bibtex, arxiv_id.
    """
    max_results = max(1, min(int(max_results), 5))
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    results = [_paper_to_dict(paper) for paper in client.results(search)]
    if not results:
        return [{"error": "No papers found. Do not invent papers."}]
    return results


@tool
def get_arxiv_paper(arxiv_id: str) -> dict:
    """
    Get one arXiv paper by id, such as '1706.03762' or '2401.12345'.

    Args:
        arxiv_id: The arXiv id or abs URL.

    Returns:
        One paper dict with title, authors, year, url, pdf_url, abstract, bibtex, arxiv_id.
    """
    arxiv_id = _clean_arxiv_id(arxiv_id)
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    papers = list(client.results(search))
    if not papers:
        return {"error": f"No paper found for id {arxiv_id}. Do not invent a paper."}
    return _paper_to_dict(papers[0])


@tool
def read_arxiv_pdf(arxiv_id: str, max_pages: int = 2) -> str:
    """
    Download an arXiv PDF and extract text from the first pages only.

    Args:
        arxiv_id: The arXiv id, such as '1706.03762'.
        max_pages: How many pages to read. Capped at 3.

    Returns:
        Extracted text from the start of the PDF, or an error string.
    """
    arxiv_id = _clean_arxiv_id(arxiv_id)
    max_pages = max(1, min(int(max_pages), 3))

    client = arxiv.Client()
    papers = list(client.results(arxiv.Search(id_list=[arxiv_id])))
    if not papers:
        return f"TOOL FAILED: no PDF found for {arxiv_id}. Do not invent content."

    pdf_url = papers[0].pdf_url
    request = urllib.request.Request(
        pdf_url,
        headers={"User-Agent": "agentlab-student/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read(15 * 1024 * 1024)
    except Exception as exc:
        return f"TOOL FAILED: could not download PDF ({exc}). Do not invent content."

    try:
        reader = PdfReader(io.BytesIO(data))
        pages = []
        for page in reader.pages[:max_pages]:
            pages.append(page.extract_text() or "")
        text = "\n".join(pages).strip()
    except Exception as exc:
        return f"TOOL FAILED: could not read PDF text ({exc}). Do not invent content."

    if not text:
        return "TOOL FAILED: PDF had no extractable text. Do not invent content."

    # Guardrail: keep the model context small
    return text[:8000]
