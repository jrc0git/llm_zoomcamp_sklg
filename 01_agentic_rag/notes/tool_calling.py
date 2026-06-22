from rag_helper import RAGBase
from ingest import build_index, load_faq_data
from dotenv import load_dotenv
import os
import json
from openai import OpenAI
import groq

load_dotenv()
llm_client = groq.Groq()

##llm_client_openai = OpenAI(
##  api_key=os.getenv('GROQ_API_KEY'),
##  base_url="https://api.groq.com/openai/v1"
##  )

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
        "query": {
          "type": "string",
          "description": "Search query text to look up in the course FAQ."
        }
      },
      "required": ["query"]
    }
  }
}



messages = [{'role': 'user', 'content': query}]


## Test api call with tool defintion.

try:
    client_response = llm_client.chat.completions.create(
                model='openai/gpt-oss-120b',
                messages=messages,
                tools=[search_tool]
            )
    
    tool_calls = client_response.choices[0].message.tool_calls

    tool_args = json.loads(tool_calls[0].function.arguments)

    print("Groq client response:")

    print(client_response.choices[0].message.to_json())

    print("tool calls:")

    print(tool_calls)

    print("tool arguments:")

    print(tool_args)

    print("search function results with tool_args:")

    search_result = search(rag, **tool_args)
    print(json.dumps(search_result, indent=2))

except Exception as e:
    print(e)


## Include tool call and tool result in messages

try:
    client_response = llm_client.chat.completions.create(
                model='openai/gpt-oss-120b',
                messages=messages,
                tools=[search_tool]
            )
    tool_calls = client_response.choices[0].message.tool_calls
    tool_args = json.loads(tool_calls[0].function.arguments)
    tool_results = search(rag, **tool_args)
    tool_results_json = json.dumps(tool_results, indent=2)
    messages.append(client_response.choices[0].message)
    messages.append(
        {
            "tool_call_id":tool_calls[0].id,
            "role":"tool",
            "name":"search",
            "content":str(tool_results_json)
        }
    )

    client_response = llm_client.chat.completions.create(
        model='openai/gpt-oss-120b',
        messages=messages,
        tools=[search_tool]
    )

    print(client_response.choices[0].message)

except Exception as e:
    print(e)

