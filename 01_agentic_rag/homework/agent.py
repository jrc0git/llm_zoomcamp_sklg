from gitsource import GithubRepositoryDataReader
from minsearch import Index
from dotenv import load_dotenv
from gitsource import chunk_documents
from rag_helper_ghnotes import RAGBase
import json
from groq import Groq

load_dotenv()

# Load data
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
files = reader.read()

documents = []

for file in files:
    doc = file.parse()
    documents.append(doc)

doc_chunk = chunk_documents(documents, size=2000, step=1000)

# Create index
index = Index(
        text_fields=['content'],
        keyword_fields=['filename']
    )
index.fit(doc_chunk)
rag = RAGBase(index=index)

# Define search
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

def search(rag_object:RAGBase, query):
  return rag_object.search(query)

# Set agent loop

llm_client = Groq()

instructions = """
You're a course teaching assistant.
You're given a question from a course student and your task is to answer it.

If you want to look up information, use the search function. 
Use as many keywords from the user question as possible when making first requests.

Make multiple searches.

Try to expand your search by using new keywords
based on the results you get from the search.

At the end, ask if there are other areas that the user wants to explore.
""".strip()


def make_call(call):
    args = json.loads(call.function.arguments)

    if call.function.name == "search":
        result = search(rag, **args)

    result_json = json.dumps(result, indent=2)

    return {
        "tool_call_id":call.id,
        "role":"tool",
        "name":"search",
        "content":result_json
    }
   



def agent_loop(instructions, question, model="llama-3.1-8b-instant") -> str:
    messages = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": question}
    ]

    it = 1

    while True:
        print(f"iteration #{it}...")
        has_function_calls = False

        response = llm_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[search_tool]
        )

        tool_calls = response.choices[0].message.tool_calls

        if tool_calls:
            messages.append(response.choices[0].message)
            for tool_call in tool_calls:
                print("function_call:", tool_call.function.name, tool_call.function.arguments)
                call_output = make_call(tool_call)
                messages.append(call_output)
                has_function_calls = True

        elif not tool_calls:
            print("ASSISTANT:")
            last_answer = response.choices[0].message.content
            print(last_answer)

        it = it + 1
        if has_function_calls == False:
            break

    return last_answer


agent_loop(instructions, question="How does the agentic loop work, and how is it different from plain RAG?")

