from __future__ import annotations

from typing_extensions import Any, Dict, TypedDict, Annotated, Literal

from langgraph.graph import StateGraph, END, START
import serpapi
from langchain_core.tools import tool
import re
import requests
from langchain_openai import ChatOpenAI
import os
from langchain_core.agents import AgentAction, AgentFinish
from agent.utils import format_rag_contexts
from agent.prompts import prompt
from agent.states import AgentState, InputState, OutputState
from agent.pinecone_db import PineconeOperations
from agent.utils import create_scratchpad
from agent.settings import config

pc = PineconeOperations()
index = pc.pinecone_load()

serpapi_params = {
    "engine": "google",
    "api_key": config["serpapi_key"]
}

@tool("fetch_arxiv")
def fetch_arxiv(arxiv_id: str):
    """Gets the abstract from an ArXiv paper given the arxiv ID.
    Useful for finding high-level context about a specific paper.
    """
    abstract_pattern = re.compile(
        r'<blockquote class="abstract mathjax">\s*<span class="descriptor">Abstract:</span>\s*(.*?)\s*</blockquote>',
        re.DOTALL
    )
    res = requests.get(
        f"https://export.arxiv.org/abs/{arxiv_id}"
    )
    re_match = abstract_pattern.search(res.text)
    return re_match.group(1)


@tool("web_search")
def web_search(query: str):
    """Finds general knowledge information using Google Search. Can
    also be used to augment 'general' knowledge to a previous
    specialist query
    """
    search = serpapi.search({
        **serpapi_params,
        "q": query,
        "num": 5,
    })
    results = search["organic_results"]
    contexts = "\n---\n".join(
        ["\n".join([x["title"], x["snippet"], x["link"]]) for x in results]
    )
    return contexts

@tool("rag_search_filter")
def rag_search_filter(query: str, arxiv_id: str, encoder, index):
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
def rag_search(query: str, encoder, index):
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


llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    temperature=0
)

tools = [
    rag_search_filter,
    rag_search,
    fetch_arxiv,
    web_search,
    final_answer
]

oracle = (
        {
            "input": lambda x: x.input,
            "chat_history": lambda x: x.chat_history,
            "scratchpad": lambda x: create_scratchpad(intermediate_steps=x.intermediate_steps)
        }
        | prompt
        | llm.bind_tools(tools, tool_choice="any")


)

def run_oracle(state: list):
    print("run oracle")
    print(f"intermediate_steps: {state.intermediate_steps}")
    out = oracle.invoke(state)
    tool_name = out.tool_calls[0]["name"]
    tool_args = out.tool_calls[0]["args"]
    action_out = AgentAction(
        tool=tool_name,
        tool_input=tool_args,
        log="TBD"
    )
    return {
        "intermediate_steps": [action_out]
    }


def router(state: list) -> Literal["rag_search_filter", "rag_search", "fetch_arxiv", "web_search", "final_answer"]:
    if isinstance(state.intermediate_steps, list):
        if state.intermediate_steps[-1].tool == "rag_search_filter":
            return "rag_search_filter"
        elif state.intermediate_steps[-1].tool == "rag_search":
            return "rag_search"
        elif state.intermediate_steps[-1].tool == "fetch_arxiv":
            return "fetch_arxiv"
        elif state.intermediate_steps[-1].tool == "web_search":
            return "web_search"
    else:
        print("Router invalid format")
        return "final_answer"


tool_str_to_funct = {
    "rag_search_filter": rag_search_filter,
    "rag_search": rag_search,
    "fetch_arxiv": fetch_arxiv,
    "web_search": web_search,
    "final_answer": final_answer
}

def run_tool(state: list):
    tool_name = state.intermediate_steps[-1].tool
    tool_args = state.intermediate_steps[-1].tool_input
    print(f"{tool_name}.invoke(input={tool_args})")
    # run tool
    out = tool_str_to_funct[tool_name].invoke(input=tool_args)
    action_out = AgentAction(
        tool=tool_name,
        tool_input=tool_args,
        log=str(out)
    )
    return {"intermediate_steps": [action_out]}


builder = StateGraph(AgentState, input=InputState, output=OutputState)

builder.add_node("oracle", run_oracle)
builder.add_node("rag_search_filter", run_tool)
builder.add_node("rag_search", run_tool)
builder.add_node("fetch_arxiv", run_tool)
builder.add_node("web_search", run_tool)
builder.add_node("final_answer", run_tool)
builder.add_edge(START, "oracle")
builder.add_conditional_edges("oracle", router)
builder.add_edge("rag_search_filter", "oracle")
builder.add_edge("rag_search", "oracle")
builder.add_edge("fetch_arxiv", "oracle")
builder.add_edge("web_search", "oracle")
builder.add_edge("final_answer", END)
graph = builder.compile(name="New Graph")
