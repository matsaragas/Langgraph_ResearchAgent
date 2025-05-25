# Set up knowledge base

from datasets import load_dateset
dataset = load_dateset("jamescalam/ai-arxiv2-semantic-chunks", split="train")


## Building a knowledge base

import os
from getpass import getpass
from semantic_router.encoders import OpenAIEncoder

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or getpass("OpenAI API key: ")

encoder = OpenAIEncoder(name="text-embedding-3-small")


# pine cone connection
# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = os.getenv("PINECONE_API_KEY") or getpass("Pinecone API key: ")

pc = Pinecone(api_key=api_key)

from pinecone import ServerlessSpec

spec = ServerlessSpec(
    cloud="aws", region="us-east-1"
)

dims = len(encoder(["some random text"])[0])
dims

import time
index_name = "gpt-4o-research-agent"

# check if index already exists (it shouldn't if this is the first time)

if index_name not in pc.list_indexes().names():
    # if it doesn't exist, create index
    pc.create_index(
        index_name,
        dimension=dims,
        metric='dotproduct',
        spec=spec
    )
    # wait for index to be initiated
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)

index = pc.Index(index_name)
time.sleep(1)
index.describe_index_stats()

## Populate our knowledge base:

from tqdm.auto import tqdm

data = dataset.to_pandas().iloc[:1000]

batch_size = 128

for i in tqdm(range(0, len(data), batch_size)):
    i_end = min(len(data), i+batch_size)
    batch = data[i:i_end].to_dict(orient="records")
    #get batch data
    metadata = [{
        "title": r["title"],
        "content": r["content"],
        "arxiv_id": r["arxiv_id"],
        "references": r["references"].to_list()
    } for r in batch]
    # generate unique ids for each chunk
    ids = [r["id"] for r in batch]
    content = [r["content"] for r in batch]
    #embed text
    embeds = encoder(content)
    # add to Pinecone
    index.upsert(vectors=zip(ids, embeds, metadata))

## Graph State

# we define a custom graph state to support our agent-oriented decision making

from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
import operator


# There are four parts to our agent state:

# 1) Input: This is the users most recent query. Usually, this is the questions
# we want the agent to answer.
#2) chat_history: we build a convesational agent and we want previous inteactions
# to provide additional context throughout our agent logic we include
# the chat history. in the agent state.
# 3) Intermediate steps: Provides a record of all steps the research a
# agent will take between the user asking a question via `input` and the
# the agent providing a final answer. This can include things like "search arxiv",
# "perform general purpose web search", etc. These intermediate steps are
# crucial to allowing the agent to follow a path of coherent actions and
# ultimately producing an informed final answer.

## Custom Tools

# We define sevela tools for this agent that will focus on initial data
# discovery, that will allow the LLM to use more tools to research more
# deeply via a variety of different routes.

# ArXiv Paper Fetch

# The `fetch_arxiv` tool: allow our agent to get a summary of a specific paper
# given an Arxiv paper ID. To dpo this we will simple send a GET request to
# arXiv and use regex to extract the paper abstract.

class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    itermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]


import requests

# test with mixtral paper
arxiv_id = "2401.04088"
res = requests.get(f"https://export.arxiv.org/abs/{arxiv_id}")
res.text

# use regex to identify the paper's abstract

import re
abstract_pattern = re.compile(
    r'<blockquote class="abstract mathjax">\s*<span class="descriptor">Abstract:</span>\s*(.*?)\s*</blockquote>',
    re.DOTALL
)

re_match = abstract_pattern.search(res.text)

print(re_match.group(1))

# Pack all of the above logic into a tool for our agent to use


from langchain_core.tools import tool

@tool("fetch_arxiv")
def fetch_arxiv(arxiv_id: str):
    """Gets the abstract from an ArXiv paper given the arxiv ID.
    Useful for finding high-level context about a specific paper.
    """
    res = requests.get(
        f"https://export.arxiv.org/abs/{arxiv_id}"
    )
    re_match = abstract_pattern.search(res.text)
    return re_match.group(1)

print(
    fetch_arxiv.invoke(input={"arxiv_id": arxiv_id})
)


## Web Search
# The web search tool will provide the agent with access to web search.
#It will be instructed to use this for more general knowledge queries

from serpapi import GoogleSearch
# https://serpapi.com/manage-api-key
serpapi_params = {
    "engine": "google",
    "api_key": os.getenv("SERPAPI_KEY") or getpass("SerpAPI key: ")
}

search = GoogleSearch({
    **serpapi_params,
    "q": "coffee"
})

results = search.get_dict()["organic_results"]
contexts = "\n---\n".join(
    ["\n".join([x["title"], x["snippet"], x["link"]]) for x in results]
)
print(contexts)


@tool("web_search")
def web_search(query: str):
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


## RAG Tool

# Provide two RAG-focused tools for our agent. The `rag_search` allows
# the agent to performa a simple RAG search for info across all indexed
# research papers. The `rag_search_filter` also searches, but within a
# specific paper which is filteres for via the `arxiv_id` parameter.

# We also define the `format_rag_contexts` function to handle the
# transformation of our Pinecone results from a JSON object to a readable
# plaintext format


from langchain_core.tools import tool

def format_rag_contexts(matches: list):
    contexts = []
    for x in matches:
        text = (
            f"Title: {x['metadata']['title']}\n"
            f"Content: {x['metadata']['content']}\n"
            f"ArXiv ID: {x['metadata']['arxiv_id']}\n"
            f"Related Papers: {x['metadata']['references']}\n"
        )
        contexts.append(text)
    context_str = "\n---\n".join(contexts)
    return context_str


@tool("rag_search_filter")
def rag_search_filter(query: str, arxiv_id: str):
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
def rag_search(query: str):
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
    return ""


## Initialize the Oracle
# The oracle LLM is our graph decision maker. It decides which
# path we should take down our graph. It functions similarly
# to an agent but is much simpler and reliable.

# Oracle Prompt

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

system_prompt = """You are the oracle, the great AI decision maker.
Given the user's query you must decide what to do with it based on the
list of tools provided to you.

If you see that a tool has been used (in the scratchpad) with a 
particular query, do NOT use that same tool with the same query again. 
Also, do NOT use any tool more than twice (ie, if the tool appears
in the scratchpad twice, do not use it again).

You should aim to collect information from a diverse range of sources 
before providing the answer to the user. Once you have collected 
plenty of information to answer the user's questions (stored in the 
scratchpad) use the final_answer tool.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("assistant", "scratchpad: {scratchpad}")
])


# Next, we must initialize our llm (gpt-4o) and then create the
# runnable pipeline of our Oracle. The runnable connects our inputs
# (the users `input` and `chat_history`) to our prompt, and our
# prompt to our llm. It is also where we bind our tools to the
# LLM and enforce function calling via `tool_choice="any"`

from langchain_core.messages import ToolCall, ToolMessage
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=os.environ["OPEN_API_KEY"],
    temperature=0
)

tools = [
    rag_search_filter,
    rag_search,
    fetch_arxiv,
    web_search,
    final_answer
]

# define a function to transform intermediate_steps from list
# of AgentAction to scratchpad string

def create_scratchpad(intermediate_steps: list[AgentAction]):
    research_steps = []
    for i, action in enumerate(intermediate_steps):
        if action.log != "TBD":
            #this was the ToolExecution
            research_steps.append(
                f"Tool: {action.tool}, input: {action.tool_input}\n"
                f"Output: {action.log}"
            )
    return "\n---\n".join(research_steps)

oracle = (
    {
    "input": lambda x: x["input"],
    "chat_history": lambda x: x["chat_history"],
    "scratchpad": lambda x: create_scratchpad(
        intermediate_steps=x["intermediate_steps"]
    ),
    }
    | prompt
    | llm.bind_tools(tools, tool_choice="any")
)




