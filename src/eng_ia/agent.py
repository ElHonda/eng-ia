from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware

from eng_ia.llm import criar_llm
from eng_ia.raciocinio import acompanhar, imprimir_resposta
from eng_ia.schemas import RespostaAgente
from eng_ia.tools import build_tools


MAX_INTERACTIONS = 20

llm = criar_llm(temperature=0.6)
nome_modelo = llm.model

system_prompt = (
    "Você é um assistente que responde em português brasileiro (pt-BR). "
    "Use arxiv e wikipedia para artigos, fatos e páginas. "
    "Use contar_historias_espaco quando o usuário quiser uma história narrada "
    "por um astronauta. "
    "Use buscar_imagens quando o usuário quiser uma foto ou imagem "
    "(a ferramenta devolve um URL). "
    "Chame só uma ferramenta por vez: primeiro a ferramenta, espere o resultado, "
    "e só então preencha RespostaAgente. "
    "Depois da ferramenta, preencha RespostaAgente na hora. Não explique o schema "
    "nem planeje os campos em texto. "
    "Com arxiv/wikipedia: mantenha titulo e autores como na fonte (não traduza "
    "nomes próprios) e escreva o resumo fiel ao conteúdo da ferramenta. "
    "Com contar_historias_espaco: titulo é um título curto da história, autores "
    "pode ser o narrador astronauta, resumo é a história, fontes inclui "
    "contar_historias_espaco. "
    "Com buscar_imagens: titulo descreve a imagem, resumo cita o que foi "
    "buscado, fontes inclui o URL retornado. "
    "Sem placeholders. Não invente campos vazios quando a ferramenta já "
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
