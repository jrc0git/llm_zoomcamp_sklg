from ingest import build_index, load_faq_data
from rag_helper import RAGBase
from dotenv import load_dotenv
import os
import json

load_dotenv()


documents = load_faq_data()
index = build_index(documents)
rag = RAGBase(index=index)

query = "Can I still join the course?"
search_result = rag.search(query)

print(json.dumps(search_result, indent=2))

## Implementamos en RAGBase la posibilidad de enviar definiciones
## de herramientas en el método encargado de comunicarse con el LLM
## a través del cliente.