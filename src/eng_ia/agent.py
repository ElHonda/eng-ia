from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from eng_ia.raciocinio import acompanhar, imprimir_resposta
from eng_ia.schemas import RespostaAgente
from eng_ia.tools import build_tools
import os


MAX_INTERACTIONS = 20
nome_modelo = os.getenv("AGENT_MODEL")

llm = ChatNVIDIA(
    model=nome_modelo,
    temperature=0.6,
)
nome_modelo = llm.model

modelo = llm._client.model
if modelo and modelo.supports_thinking:
    llm = llm.with_thinking_mode(enabled=True)

system_prompt = (
    "Você é um assistente que responde em português brasileiro (pt-BR). "
    "Use arxiv e wikipedia. Chame só uma ferramenta por vez: primeiro arxiv "
    "ou wikipedia, espere o resultado, e só então preencha RespostaAgente. "
    "Mantenha titulo e autores como na fonte (não traduza nomes próprios). "
    "Escreva o resumo em português brasileiro, fiel ao conteúdo da ferramenta, "
    "sem placeholders. Não invente campos vazios quando a ferramenta já "
    "trouxe os dados."
)

agent = create_agent(
    model=llm,
    tools=build_tools(),
    system_prompt=system_prompt,
    response_format=RespostaAgente,
    middleware=[
        ModelCallLimitMiddleware(run_limit=MAX_INTERACTIONS, exit_behavior="end"),
    ],
)


def run(pergunta: str | None = None) -> None:
    print(f"Modelo: {nome_modelo}")
    pergunta = pergunta or "Sobre o que é o artigo '2409.15934'?"
    estado = acompanhar(
        agent,
        pergunta,
        {"recursion_limit": MAX_INTERACTIONS * 4},
    )
    imprimir_resposta(estado)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Agent eng-ia")
    parser.add_argument("pergunta", nargs="?", help="Pergunta em português")
    args = parser.parse_args()
    run(args.pergunta)
