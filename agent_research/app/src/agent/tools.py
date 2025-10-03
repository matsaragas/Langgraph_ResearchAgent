from serpapi import GoogleSearch
from langchain_core.tools import tool
from utils import format_rag_contexts
from semantic_router.encoders import OpenAIEncoder


@tool("web_search")
def web_search(query: str, serpapi_params: dict):
    """Finds general knowledge information using Google Search. Can
    also be used to augment 'general' knowledge to a previous
    specialist query
    """
    search = GoogleSearch({
        **serpapi_params,
        "q": query,
        "num": 5,
    })
    results = search.get_dict()["organic_results"]
    contexts = "\n---\n".join(
        ["\n".join([x["title"], x["snippet"], x["link"]]) for x in results]
    )
    return contexts


@tool("rag_search_filter")
def rag_search_filter(query: str, arxiv_id: str, encoder: OpenAIEncoder, index):
    """Finds information from our ArXiv database using a natural
    language query and a specific ArXiv ID. Allows us to learn
    more details about a specific paper.
    """
    xq = encoder([query])
    xc = index.query(vector=xq, top_k=6, include_metadata=True,
                     filter={"arxiv_id": arxiv_id})
    context_str = format_rag_contexts(xc["matches"])
    return context_str


@tool("rag_search")
def rag_search(query: str, encoder: OpenAIEncoder, index):
    """Finds specialist information on AI using natural language query."""
    xq = encoder([query])
    xc = index.query(vector=xq, top_k=2, include_metadata=True)
    context_str = format_rag_contexts(xc["matches"])
    return context_str


@tool("final_answer")
def final_answer(
        introduction: str,
        research_steps: str,
        main_body: str,
        conclusion: str,
        sources: str
):
    """Return a natural language response to the user in the form
    of a research report. There are several sections to this report,
    those are:
    - `introduction`: a short paragraph introducing the user's questions
    and the topic we are researching.
    - `research_steps`: a few bullet points explaining the steps that were
    taken to research your report
    - `main_body`: this is where the bulk of high quality and concise
    information that answer the user's question belongs. It is 3-4
    paragraphs long in length.
    - `conclusion`: this is a short single paragraph conclusion providing a
    concise but sophisticated view on what was found.
    - `sources`: a bulletpoint list provided detailed sources for all
    information referenced during the research process
    """
    if type(research_steps) is list:
        research_steps = "\n".join([f"- {r}" for r in research_steps])
    if type(sources) in list:
        sources = "\n".join([f"- {s}" for s in sources])
    return ""