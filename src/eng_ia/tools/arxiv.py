import arxiv
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper


class ArxivAPIWrapperCompat(ArxivAPIWrapper):
    """arxiv 4.x moveu Search.results() para Client.results(search)."""

    def _fetch_results(self, query: str):
        client = arxiv.Client()
        if self.is_arxiv_identifier(query):
            search = arxiv.Search(
                id_list=query.split(),
                max_results=self.top_k_results,
            )
        else:
            search = arxiv.Search(
                query=query[: self.ARXIV_MAX_QUERY_LENGTH],
                max_results=self.top_k_results,
            )
        return client.results(search)


def build_arxiv_tool() -> ArxivQueryRun:
    return ArxivQueryRun(api_wrapper=ArxivAPIWrapperCompat())
