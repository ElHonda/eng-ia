import os
import warnings

from langchain_nvidia_ai_endpoints import ChatNVIDIA

_TIMEOUT_PADRAO = 180.0

warnings.filterwarnings(
    "ignore",
    message=r"Found .+ in available_models, but type is unknown",
)
warnings.filterwarnings(
    "ignore",
    message=r"Model '.+' is not known to support tools",
)


def criar_llm(
    *,
    temperature: float,
    max_tokens: int | None = None,
    pensar: bool = True,
) -> ChatNVIDIA:
    """ChatNVIDIA com o modelo e o timeout do ambiente.

    A chave NVIDIA_API_KEY já é lida pelo SDK no ambiente.
    """
    kwargs: dict = {}
    if max_tokens is not None:
        kwargs["max_completion_tokens"] = max_tokens

    llm = ChatNVIDIA(
        model=os.getenv("AGENT_MODEL"),
        temperature=temperature,
        timeout=float(os.getenv("NVIDIA_TIMEOUT", _TIMEOUT_PADRAO)),
        **kwargs,
    )

    if pensar:
        modelo = llm._client.model
        if modelo and modelo.supports_thinking:
            llm = llm.with_thinking_mode(enabled=True)

    return llm
