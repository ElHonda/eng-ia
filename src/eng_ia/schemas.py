import re

from pydantic import BaseModel, Field, field_validator, model_validator

_PLACEHOLDER = re.compile(r"\[.*?\]|assunto do artigo", re.IGNORECASE)


class RespostaAgente(BaseModel):
    """Formato estruturado da resposta do assistente."""

    titulo: str = Field(
        min_length=5,
        description=(
            "Título oficial do artigo ou da página, exatamente como retornado "
            "pela ferramenta. Não use títulos genéricos como 'Artigo 2409.15934'."
        ),
    )
    autores: list[str] = Field(
        description=(
            "Nomes completos dos autores extraídos da ferramenta. "
            "Em artigo científico (arxiv), inclua todos os autores. "
            "Só deixe vazio se a fonte realmente não tiver autores."
        ),
    )
    resumo: str = Field(
        min_length=80,
        description=(
            "Resumo em português brasileiro, fiel ao Summary/content "
            "retornado pela ferramenta. Sem placeholders como "
            "'[assunto do artigo]'."
        ),
    )
    fontes: list[str] = Field(
        min_length=1,
        description="Fontes consultadas (wikipedia, arxiv, IDs ou URLs)",
    )

    @field_validator("titulo", "resumo")
    @classmethod
    def sem_placeholder(cls, value: str) -> str:
        if _PLACEHOLDER.search(value):
            raise ValueError(
                "Não use placeholders. Copie título e resumo reais da ferramenta."
            )
        return value.strip()

    @field_validator("autores")
    @classmethod
    def limpar_autores(cls, value: list[str]) -> list[str]:
        return [autor.strip() for autor in value if autor.strip()]

    @model_validator(mode="after")
    def artigo_exige_autores(self) -> "RespostaAgente":
        fontes = " ".join(self.fontes).lower()
        if "arxiv" in fontes and not self.autores:
            raise ValueError(
                "Fonte arxiv exige autores extraídos da ferramenta. "
                "Preencha autores com os nomes retornados no campo Authors."
            )
        return self
