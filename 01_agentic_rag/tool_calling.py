from rag_helper import RAGBase
from ingest import build_index, load_faq_data
from dotenv import load_dotenv
import os
import json
from openai import OpenAI
import groq

load_dotenv()
llm_client = groq.Groq()

## llm_client = OpenAI(
##            api_key=os.getenv('GROQ_API_KEY'),
##            base_url="https://api.groq.com/openai/v1"
##            )

documents = load_faq_data()
index = build_index(documents)
rag = RAGBase(index=index, course='llm-zoomcamp')

def search(rag_object:RAGBase, query):
    return rag_object.search(query)

query = "Can I still join the course?"

## Search function defintion
search_tool =   {
  "type": "function",
  "function": {
    "name": "search",
    "description": "Search the FAQ database for entries matching the given query.",
    "parameters": {
      "type": "object",
      "properties": {
        "expression": {
          "type": "string",
          "description": "Search query text to look up in the course FAQ."
        }
      },
      "required": ["expression"]
    }
  }
}


messages = {'role': 'user', 'content': query}


## Test api call with tool defintion.

try:
    client_response = llm_client.chat.completions.create(
                model='openai/gpt-oss-120b',
                messages=[messages],
                tools=[search_tool]
            )

    print(client_response.choices[0].message.tool_calls)

except Exception as e:
    print(e)
