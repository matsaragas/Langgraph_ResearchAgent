from serpapi import GoogleSearch

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