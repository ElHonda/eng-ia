from langchain.tools import BaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from eng_ia.llm import criar_llm

INSTRUCTIONS = (
    "Você é um astronauta aposentado que viveu muitas experiências no espaço. "
    "Conte ao usuário uma história sobre o que ele deseja ouvir: vida em órbita, "
    "missões, treino, lançamentos, a Terra vista da estação. "
    "Responda em português brasileiro, em tom narrativo e pessoal. "
    "Escreva 3 a 5 parágrafos curtos, com começo, meio e fim. "
    "A última frase precisa terminar; nunca corte no meio da palavra. "
    "Você ainda é humano: não fale de experiências que um ser humano não poderia "
    "ter vivido. Se o pedido sair disso, recuse com humor e ofereça uma história "
    "humana equivalente."
)


class HistoriasInput(BaseModel):
    query: str = Field(
        description="Tema da história que o astronauta deve contar",
    )


_cadeia = None


def _cadeia_astronauta():
    global _cadeia
    if _cadeia is None:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", INSTRUCTIONS),
                ("human", "{input}"),
            ]
        )
        _cadeia = (
            prompt
            | criar_llm(temperature=0.5, max_tokens=2048, pensar=False)
            | StrOutputParser()
        )
    return _cadeia


class HistoriasDoEspacoTool(BaseTool):
    name: str = "contar_historias_espaco"
    description: str = (
        "Conversa com um astronauta aposentado que conta histórias sobre o "
        "espaço. Use quando o usuário quiser uma narrativa, não um artigo "
        "científico nem uma imagem."
    )
    args_schema: type[BaseModel] = HistoriasInput

    def _run(self, query: str) -> str:
        return _cadeia_astronauta().invoke({"input": query})


def build_historias_tool() -> HistoriasDoEspacoTool:
    return HistoriasDoEspacoTool()
