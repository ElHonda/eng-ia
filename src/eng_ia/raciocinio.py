from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from eng_ia.schemas import RespostaAgente


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
    if thought:
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
    print(_texto(mensagem.content).rstrip())
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


def imprimir_resposta(estado: dict) -> None:
    resultado = estado.get("structured_response")
    if isinstance(resultado, RespostaAgente):
        print(f"Final Answer: {resultado.resumo}")
        print()
        print("> Finished chain.")
        print("Resposta final:")
        print(resultado.model_dump_json(indent=2, ensure_ascii=False))
        return

    for mensagem in reversed(estado.get("messages") or []):
        if isinstance(mensagem, AIMessage):
            texto = _thought_do_modelo(mensagem)
            visivel = _texto(mensagem.content).strip()
            if "</think>" in visivel:
                visivel = visivel.split("</think>", 1)[-1].strip()
            final = visivel or texto
            if final and not mensagem.tool_calls:
                print(f"Final Answer: {final}")
                print()
                print("> Finished chain.")
                print(f"Resposta final: {final}")
                return

    print("> Finished chain.")
    print("Resposta final: não foi possível montar a resposta estruturada.")
