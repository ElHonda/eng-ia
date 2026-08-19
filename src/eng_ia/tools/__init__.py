from langchain_core.tools import BaseTool

from eng_ia.tools.arxiv import build_arxiv_tool
from eng_ia.tools.wikipedia import build_wikipedia_tool


def build_tools() -> list[BaseTool]:
    return [build_arxiv_tool(), build_wikipedia_tool()]
