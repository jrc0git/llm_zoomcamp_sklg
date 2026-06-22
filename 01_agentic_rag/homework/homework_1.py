from gitsource import GithubRepositoryDataReader
import json
from minsearch import Index
from rag_helper_ghnotes import RAGBase
from dotenv import load_dotenv
from gitsource import chunk_documents
from agent import agent_loop

load_dotenv()
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

## Example of document
print(json.dumps(documents[0],indent=2))

## Q1: How many lesson pages are in the dataset?
print("Q1: How many lesson pages are in the dataset?")
print(len(documents))

## Q2: Index the documents with minsearch - make content a text field and filename a keyword field. Then search with this query:
## How does the agentic loop keep calling the model until it stops?

print("""Q2: Index the documents with minsearch - make content a text field and filename a keyword field. Then search with this query:
## How does the agentic loop keep calling the model until it stops?""".strip())

query = "How does the agentic loop keep calling the model until it stops?"

index = Index(
        text_fields=['content'],
        keyword_fields=['filename']
    )
index.fit(documents)

results = index.search(query)

print(results[0].get('filename'))


## Q3: Build a RAG over the index from Q2 and answer the query: How does the agentic loop keep calling the model until it stops? 
## Use gpt-5.4-mini. How many input (prompt) tokens did we send to the model for this request?

print("""Q3: Build a RAG over the index from Q2 and answer the query: How does the agentic loop keep calling the model until it stops? 
Use gpt-5.4-mini. How many input (prompt) tokens did we send to the model for this request?""".strip())

rag = RAGBase(index=index)
query = "How does the agentic loop keep calling the model until it stops?"
rag_answer = rag.rag(query)
answer_content, input_tokens = rag_answer
print(answer_content)
print(input_tokens)



## Q4: How many chunks do you get? (after applying chunking to documents)

print("Q4: How many chunks do you get? (after applying chunking to documents)")

doc_chunk = chunk_documents(documents, size=2000, step=1000)
print(len(doc_chunk))


## Q5. RAG with chunking
## Index the chunks from Q4 (same as before: content as a text field, filename as a keyword field), point your RAG at the chunk index, and answer the same query again - reading the input tokens the same way as in Q3.
## Compare the input tokens with Q3. How many fewer input tokens does the chunked version send?

print("""Q5. RAG with chunking
Index the chunks from Q4 (same as before: content as a text field, filename as a keyword field), point your RAG at the chunk index, and answer the same query again - reading the input tokens the same way as in Q3.
Compare the input tokens with Q3. How many fewer input tokens does the chunked version send?""".strip())

index = Index(
        text_fields=['content'],
        keyword_fields=['filename']
    )
index.fit(doc_chunk)

rag = RAGBase(index=index)
query = "How does the agentic loop keep calling the model until it stops?"
rag_answer = rag.rag(query)
answer_content, input_tokens = rag_answer
print(answer_content)
print(input_tokens)

print("Dif prompt tokens: 7184/2367=~3.03")


##Q6: Build an agent with your search tool and run it (with toyaikit, the same way as in the ToyAIKit lesson). Use these instructions for the agent (they nudge it to search a few times):

## You're a course teaching assistant. Answer the student's question using the search tool. Make multiple searches with different keywords before answering.

## Ask it:

##How does the agentic loop work, and how is it different from plain RAG?

##The agent decides on its own when to search and when to answer. Count how many times it called the search tool.

##How many times did the agent call search?

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

agent_loop(instructions, question="How does the agentic loop work, and how is it different from plain RAG?")

## Model used: llama-3.1-8b-instant
## Only needs 1 tool call to generate an answer.