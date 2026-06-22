import os
from groq import Groq

INSTRUCTIONS = '''
You are a helpful assistant. Your task is to answer user questions
based on the provided context. If the answer is not found in the context,
respond with "I don't know."
'''

PROMPT_TEMPLATE = '''
QUESTION: {question}

CONTEXT:
{context}
'''.strip()


class RAGBase:

    def __init__(
        self,
        index,
        llm_client=None,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        model='openai/gpt-oss-120b'
    ):
        self.index = index
        if llm_client is None:
            self.llm_client = Groq()
        else:
            self.llm_client = llm_client
        self.instructions = instructions
        self.filename = None
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        filter_dict = None if self.filename is None else {
            "filename":self.filename
        } 
        return self.index.search(
            query,
            num_results=num_results,
            filter_dict=filter_dict
        )

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append('Filename:' + doc['filename'])
            lines.append('Content: ' + doc['content'])
            lines.append('')

        return '\n'.join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, prompt):
        input_messages = [
            {'role': 'system', 'content': self.instructions},
            {'role': 'user', 'content': prompt}
        ]

        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=input_messages
        )

        response_content = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens

        return response_content, input_tokens


    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer
    
