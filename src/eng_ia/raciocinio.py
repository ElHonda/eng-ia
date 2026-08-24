from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from eng_ia.schemas import RespostaAgente

_HISTORIA = "contar_historias_espaco"
_IMAGEM = "buscar_imagens"
_PLANEJAMENTO = (
    "RespostaAgente",
    "We need to produce",
    "We need to fill",
    "We need to respond",
    "Make RespostaAgente",
    "structured_response",
    "titulo, autores, resumo",
)


def _texto(conteudo: object) -> str:
    if isinstance(conteudo, str):
        return conteudo
    return str(conteudo)


def _action_input(tool_call: dict) -> str:
    args = tool_call.get("args") or {}
    if "query" in args:
        return str(args["query"])
    if args:
        return str(args)
    return ""


def _eh_planejamento(texto: str) -> bool:
    return any(marca in texto for marca in _PLANEJAMENTO)


def _thought_do_modelo(mensagem: AIMessage) -> str:
    extra = mensagem.additional_kwargs or {}
    for chave in ("reasoning_content", "reasoning"):
        valor = extra.get(chave)
        if isinstance(valor, str) and valor.strip():
            return valor.strip()

    for bloco in getattr(mensagem, "content_blocks", []) or []:
        if not isinstance(bloco, dict):
            continue
        if bloco.get("type") not in {"reasoning", "thinking"}:
            continue
        texto = bloco.get("reasoning") or bloco.get("text") or ""
        if str(texto).strip():
            return str(texto).strip()

    conteudo = _texto(mensagem.content).strip()
    if "</think>" in conteudo:
        if "<think>" in conteudo:
            inicio = conteudo.find("<think>") + len("<think>")
            fim = conteudo.find("</think>")
            return conteudo[inicio:fim].strip()
        return conteudo.split("</think>", 1)[0].strip()
    return conteudo


def _imprimir_thought_e_actions(mensagem: AIMessage) -> None:
    thought = _thought_do_modelo(mensagem)
    if thought and not _eh_planejamento(thought):
        print(f"Thought: {thought}")

    for tool_call in mensagem.tool_calls or []:
        nome = tool_call.get("name", "ferramenta")
        if nome == "RespostaAgente":
            continue
        print(f"Action: {nome}")
        print(f"Action Input: {_action_input(tool_call)}")


def _imprimir_observation(mensagem: ToolMessage) -> None:
    if mensagem.name == "RespostaAgente":
        return
    texto = _texto(mensagem.content).rstrip()
    if mensagem.name == _HISTORIA:
        print("História:")
        print()
        print(texto)
        print()
        return
    print(texto)
    print()


def _imprimir_mensagens(mensagens: list[BaseMessage], vistos: set[str]) -> None:
    for mensagem in mensagens:
        mid = getattr(mensagem, "id", None) or str(id(mensagem))
        if mid in vistos:
            continue
        vistos.add(mid)
        if isinstance(mensagem, AIMessage):
            _imprimir_thought_e_actions(mensagem)
        elif isinstance(mensagem, ToolMessage):
            _imprimir_observation(mensagem)


def acompanhar(agent, pergunta: str, config: dict | None = None) -> dict:
    """Executa o agent imprimindo o raciocínio no formato Thought / Action / Observation."""
    print("> Entering new agent chain...")
    vistos: set[str] = set()
    estado_final: dict = {}

    for modo, evento in agent.stream(
        {"messages": [{"role": "user", "content": pergunta}]},
        config or {},
        stream_mode=["updates", "values"],
    ):
        if modo == "values":
            estado_final = evento
            continue
        if modo != "updates" or not isinstance(evento, dict):
            continue
        for update in evento.values():
            if isinstance(update, dict) and "messages" in update:
                _imprimir_mensagens(update["messages"], vistos)

    return estado_final


def _observacao(estado: dict, nome: str) -> str | None:
    for mensagem in reversed(estado.get("messages") or []):
        if isinstance(mensagem, ToolMessage) and mensagem.name == nome:
            texto = _texto(mensagem.content).strip()
            if texto:
                return texto
    return None


def _imprimir_bloco_final(titulo: str, corpo: str, fontes: list[str] | None = None) -> None:
    print("Final Answer:")
    print()
    print("=" * 80)
    print(titulo)
    print()
    print(corpo.rstrip())
    if fontes:
        print()
        print("Fonte: " + ", ".join(fontes))
    print()
    print("=" * 80)
    print("> Finished chain.")


def _imprimir_resposta_agente(resultado: RespostaAgente) -> None:
    fontes = " ".join(resultado.fontes).lower()
    if _HISTORIA in fontes:
        _imprimir_bloco_final(resultado.titulo, resultado.resumo, resultado.fontes)
        return
    if _IMAGEM in fontes or any(fonte.startswith("http") for fonte in resultado.fontes):
        _imprimir_bloco_final(resultado.titulo, resultado.resumo, resultado.fontes)
        return

    print("Final Answer:")
    print()
    print(f"Título: {resultado.titulo}")
    if resultado.autores:
        print(f"Autores: {', '.join(resultado.autores)}")
    print()
    print(resultado.resumo.rstrip())
    print()
    print("Fontes: " + ", ".join(resultado.fontes))
    print()
    print("> Finished chain.")


def imprimir_resposta(estado: dict) -> None:
    resultado = estado.get("structured_response")
    if isinstance(resultado, RespostaAgente):
        _imprimir_resposta_agente(resultado)
        return

    historia = _observacao(estado, _HISTORIA)
    if historia:
        _imprimir_bloco_final("História do astronauta", historia, [_HISTORIA])
        return

    imagem = _observacao(estado, _IMAGEM)
    if imagem:
        _imprimir_bloco_final("Imagem", imagem, [_IMAGEM])
        return

    for mensagem in reversed(estado.get("messages") or []):
        if not isinstance(mensagem, AIMessage) or mensagem.tool_calls:
            continue
        visivel = _texto(mensagem.content).strip()
        if "</think>" in visivel:
            visivel = visivel.split("</think>", 1)[-1].strip()
        if visivel and not _eh_planejamento(visivel):
            print(f"Final Answer: {visivel}")
            print()
            print("> Finished chain.")
            return

    print("> Finished chain.")
    print("Resposta final: não foi possível montar a resposta estruturada.")
