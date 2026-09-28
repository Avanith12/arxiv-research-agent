import io
import re
import urllib.request
from datetime import datetime

import arxiv
from pypdf import PdfReader
from smolagents import tool

MAX_PAPERS = 5
NEWEST_POOL_SIZE = 25
MAX_PDF_PAGES = 3
MAX_PDF_CHARS = 8000

# Every paper the agent finds is kept here so the UI can export BibTeX.
SAVED_PAPERS = {}


def clear_saved_papers():
    SAVED_PAPERS.clear()


def _clean_arxiv_id(arxiv_id: str) -> str:
    value = (arxiv_id or "").strip()
    for prefix in (
        "https://arxiv.org/abs/",
        "http://arxiv.org/abs/",
        "https://arxiv.org/pdf/",
        "http://arxiv.org/pdf/",
        "arxiv:",
        "arXiv:",
    ):
        value = value.replace(prefix, "")
    return value.replace(".pdf", "").strip()


def _base_id(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", _clean_arxiv_id(arxiv_id))


def _paper_to_dict(paper) -> dict:
    arxiv_id = paper.get_short_id()
    cite_key = _base_id(arxiv_id).replace("/", "_")
    authors = ", ".join(author.name for author in paper.authors)
    bibtex_authors = " and ".join(author.name for author in paper.authors)
    bibtex = (
        f"@article{{{cite_key},\n"
        f"  title = {{{paper.title}}},\n"
        f"  author = {{{bibtex_authors}}},\n"
        f"  year = {{{paper.published.year}}},\n"
        f"  eprint = {{{cite_key}}},\n"
        f"  archivePrefix = {{arXiv}},\n"
        f"  primaryClass = {{{paper.primary_category}}},\n"
        f"  url = {{{paper.entry_id}}}\n"
        f"}}"
    )
    result = {
        "arxiv_id": arxiv_id,
        "title": paper.title,
        "authors": authors,
        "year": paper.published.year,
        "published": paper.published.strftime("%Y-%m-%d"),
        "category": paper.primary_category,
        "url": paper.entry_id,
        "pdf_url": paper.pdf_url,
        "abstract": paper.summary,
        "bibtex": bibtex,
    }
    SAVED_PAPERS[cite_key] = result
    return result


def _fetch_by_ids(ids: list) -> list:
    client = arxiv.Client()
    return list(client.results(arxiv.Search(id_list=ids)))


def _build_query(query: str, author: str, category: str, since_year: int) -> str:
    parts = []
    # Require every keyword; plain text makes arXiv match any single word.
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\.]*", query)
    if words:
        parts.append("(" + " AND ".join(f"all:{word}" for word in words) + ")")
    if author.strip():
        parts.append(f'au:"{author.strip().replace(chr(34), "")}"')
    if category.strip():
        parts.append(f"cat:{category.strip()}")
    if since_year:
        parts.append(f"submittedDate:[{since_year}01010000 TO 209912312359]")
    return " AND ".join(parts)


@tool
def search_arxiv(
    query: str = "",
    max_results: int = 3,
    newest_first: bool = False,
    since_year: int = 0,
    category: str = "",
    author: str = "",
) -> list:
    """
    Search arXiv for academic papers. Combine any of the filters.

    Args:
        query: Topic keywords, e.g. 'agentic AI safety'. Can be empty if author is given.
        max_results: How many papers to return. Capped at 5.
        newest_first: True to sort by newest submission date instead of relevance.
        since_year: Only papers submitted in this year or later, e.g. 2024. Use 0 for no filter.
        category: arXiv category such as 'cs.AI', 'cs.LG', 'cs.CL', 'stat.ML'. Empty for any.
        author: Author name, e.g. 'Yann LeCun' or 'LeCun'. Empty for any.

    Returns:
        A list of paper dicts with arxiv_id, title, authors, year, published, category, url, pdf_url, abstract, bibtex.
    """
    max_results = max(1, min(int(max_results), MAX_PAPERS))
    since_year = int(since_year or 0)
    if since_year and not (1991 <= since_year <= datetime.now().year):
        return [{"error": f"since_year {since_year} is not a valid year."}]

    search_query = _build_query(query, author, category, since_year)
    if not search_query:
        return [{"error": "Give a topic, an author, or a category to search."}]

    # Sorting all of arXiv by date returns loosely related papers for topic searches, so for
    # newest_first take the most relevant pool and sort that pool by date. Author or
    # category searches with no topic have no relevance signal, so sort by date directly.
    has_topic = bool(re.search(r"[A-Za-z0-9]", query))
    if newest_first and not has_topic:
        pool_size, sort_by = max_results, arxiv.SortCriterion.SubmittedDate
    elif newest_first:
        pool_size, sort_by = NEWEST_POOL_SIZE, arxiv.SortCriterion.Relevance
    else:
        pool_size, sort_by = max_results, arxiv.SortCriterion.Relevance

    client = arxiv.Client()
    search = arxiv.Search(query=search_query, max_results=pool_size, sort_by=sort_by)
    try:
        papers = list(client.results(search))
    except Exception as exc:
        return [{"error": f"TOOL FAILED: arXiv search error ({exc}). Do not invent papers."}]

    if newest_first:
        papers.sort(key=lambda paper: paper.published, reverse=True)
    papers = papers[:max_results]

    if not papers:
        return [{"error": "No papers found. Do not invent papers."}]
    return [_paper_to_dict(paper) for paper in papers]


@tool
def get_arxiv_paper(arxiv_id: str) -> dict:
    """
    Get one arXiv paper by id, such as '1706.03762' or '2401.12345'.

    Args:
        arxiv_id: The arXiv id or abs URL.

    Returns:
        One paper dict with arxiv_id, title, authors, year, published, category, url, pdf_url, abstract, bibtex.
    """
    arxiv_id = _clean_arxiv_id(arxiv_id)
    papers = _fetch_by_ids([arxiv_id])
    if not papers:
        return {"error": f"No paper found for id {arxiv_id}. Do not invent a paper."}
    return _paper_to_dict(papers[0])


@tool
def compare_papers(arxiv_id_1: str, arxiv_id_2: str) -> dict:
    """
    Fetch two arXiv papers so they can be compared side by side.

    Args:
        arxiv_id_1: The first arXiv id, e.g. '1706.03762'.
        arxiv_id_2: The second arXiv id, e.g. '1810.04805'.

    Returns:
        A dict with 'paper_1' and 'paper_2', each a full paper dict, or an 'error' key.
    """
    id_1 = _base_id(arxiv_id_1)
    id_2 = _base_id(arxiv_id_2)
    if id_1 == id_2:
        return {"error": "Both ids are the same paper. Give two different papers."}

    found = {_base_id(p.get_short_id()): p for p in _fetch_by_ids([id_1, id_2])}
    paper_1 = found.get(id_1)
    paper_2 = found.get(id_2)
    missing = [pid for pid, p in ((id_1, paper_1), (id_2, paper_2)) if p is None]
    if missing:
        return {"error": f"No paper found for {', '.join(missing)}. Do not invent a paper."}

    return {"paper_1": _paper_to_dict(paper_1), "paper_2": _paper_to_dict(paper_2)}


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
    max_pages = max(1, min(int(max_pages), MAX_PDF_PAGES))

    papers = _fetch_by_ids([arxiv_id])
    if not papers:
        return f"TOOL FAILED: no PDF found for {arxiv_id}. Do not invent content."

    request = urllib.request.Request(
        papers[0].pdf_url,
        headers={"User-Agent": "arxiv-research-agent/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read(15 * 1024 * 1024)
    except Exception as exc:
        return f"TOOL FAILED: could not download PDF ({exc}). Do not invent content."

    try:
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages[:max_pages]).strip()
    except Exception as exc:
        return f"TOOL FAILED: could not read PDF text ({exc}). Do not invent content."

    if not text:
        return "TOOL FAILED: PDF had no extractable text. Do not invent content."
    return text[:MAX_PDF_CHARS]
