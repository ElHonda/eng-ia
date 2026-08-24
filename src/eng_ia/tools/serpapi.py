import os

from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from serpapi import GoogleSearch


class SerpapiInput(BaseModel):
    query: str = Field(description="Tema da imagem a buscar na web")


class SerpapiSearchTool(BaseTool):
    name: str = "buscar_imagens"
    description: str = (
        "Busca imagens na internet via SerpApi (Google Imagens). "
        "A entrada é o tema da busca; a saída é o link da primeira imagem. "
        "Use quando o usuário quiser uma foto, ilustração ou imagem."
    )
    args_schema: type[BaseModel] = SerpapiInput

    def _run(self, query: str) -> str:
        chave = os.getenv("SERPAPI_API_KEY")
        if not chave:
            return (
                "SERPAPI_API_KEY não está definida. É a chave do SerpApi "
                "(serpapi.com), não a da NVIDIA."
            )
        busca = GoogleSearch(
            {
                "engine": "google",
                "q": query,
                "tbm": "isch",
                "num": 10,
                "hl": "pt",
                "gl": "br",
                "api_key": chave,
            }
        )
        resultados = busca.get_dict()
        imagens = resultados.get("images_results") or []
        if not imagens:
            erro = resultados.get("error")
            if erro:
                return f"SerpApi retornou erro: {erro}"
            return f"Nenhuma imagem encontrada para: {query}"
        return imagens[0].get("original") or imagens[0].get("link") or (
            f"Nenhum link de imagem em: {query}"
        )


def build_serpapi_tool() -> SerpapiSearchTool:
    return SerpapiSearchTool()
