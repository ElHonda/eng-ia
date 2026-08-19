from langchain.tools import tool
from langchain.tools import BaseTool
from langchain_core.tools import StructuredTool

# @tool
def size_text(text: str, max_len: int) -> str:
  """Calcula o tamanho de um texto e verifica se ultrapassa o máximo permitido.

  Args:
    text: Texto a ser medido
    max_len: Tamanho máximo permitido
  """
  if len(text) > max_len:
    return "Excedeu o tamanho máximo."
  
  return "O tamanho é permitido"
    

## criação da tool usando o dataclass StructuredTool
# from eng_ia.tools.length import size_text;
# from langchain.tools import tool
# from langchain_core.tools import StructuredTool
# desc = "Calcula o tamanho de um texto e verifica se ultrapassa o máximo permitido."
# calc_size = StructuredTool.from_function(func=size_text, name="Calcular Tamanho", description=desc)
# text = "O cinema no Brasil é uma expressão vibrante e diversificada da cultura"
# result = calc_size.run({"text": text, "max_len": 20})
# print(result)

class LengthTool(BaseTool):
  name: str = "length_text"
  description: str = "Usada para calcular o tamanho de um texto e verifica se ultrapassa o máximo permitido"

  def _run(self, text: str, max_len: int) -> str:
    """Use the tool."""
    size = len(text)

    if len(text) > max_len:
      return "Excedeu o tamanho máximo."
  
    return "O tamanho é permitido"

# from eng_ia.tools.length import LengthTool;
# length_text = LengthTool()
# text = "O cinema no Brasil é uma expressão vibrante e diversificada da cultura"
# result = length_text.invoke({"text": text, "max_len": 20})
# print(result)
