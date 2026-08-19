from json import JSONDecodeError
from typing import Any, Optional

import wikipedia as wikipedia_lib
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities.wikipedia import WIKIPEDIA_MAX_QUERY_LENGTH
from requests.exceptions import RequestException

USER_AGENT = "eng-ia/0.2 (educational project)"

_FALHAS_DE_PAGINA = (
    wikipedia_lib.exceptions.WikipediaException,
    JSONDecodeError,
    RequestException,
    KeyError,
    ValueError,
)


class WikipediaAPIWrapperCompat(WikipediaAPIWrapper):
    """Pula páginas vazias ou com resposta inválida da API, sem derrubar o agent."""

    def _fetch_page(self, page: str) -> Optional[Any]:
        try:
            return self.wiki_client.page(title=page, auto_suggest=False)
        except _FALHAS_DE_PAGINA:
            return None

    def _formatted_page_summary(
        self, page_title: str, wiki_page: Any
    ) -> Optional[str]:
        try:
            summary = (wiki_page.summary or "").strip()
        except _FALHAS_DE_PAGINA:
            return None
        if not summary:
            return None
        return f"Page: {page_title}\nSummary: {summary}"

    def run(self, query: str) -> str:
        try:
            page_titles = self.wiki_client.search(
                query[:WIKIPEDIA_MAX_QUERY_LENGTH],
                results=self.top_k_results,
            )
        except _FALHAS_DE_PAGINA:
            return "No good Wikipedia Search Result was found"

        summaries = []
        for page_title in page_titles[: self.top_k_results]:
            wiki_page = self._fetch_page(page_title)
            if not wiki_page:
                continue
            summary = self._formatted_page_summary(page_title, wiki_page)
            if summary:
                summaries.append(summary)

        if not summaries:
            return "No good Wikipedia Search Result was found"
        return "\n\n".join(summaries)[: self.doc_content_chars_max]


def build_wikipedia_tool() -> WikipediaQueryRun:
    wikipedia_lib.set_user_agent(USER_AGENT)
    return WikipediaQueryRun(api_wrapper=WikipediaAPIWrapperCompat(lang="pt"))
