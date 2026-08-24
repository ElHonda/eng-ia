from langchain_core.tools import BaseTool

from eng_ia.tools.arxiv import build_arxiv_tool
from eng_ia.tools.historias import build_historias_tool
from eng_ia.tools.serpapi import build_serpapi_tool
from eng_ia.tools.wikipedia import build_wikipedia_tool
from langchain_community.tools import YouTubeSearchTool


def build_tools() -> list[BaseTool]:
    return [
        build_arxiv_tool(),
        build_wikipedia_tool(),
        build_historias_tool(),
        build_serpapi_tool(),
        YouTubeSearchTool()
    ]
