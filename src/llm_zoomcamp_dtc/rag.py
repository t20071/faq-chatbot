"""Single-turn RAG: search the FAQ, build a prompt, ask the LLM.

The multi-turn version used by the deployed app lives in chatbot/faqbot/.
"""

from openai import OpenAI

from llm_zoomcamp_dtc.config import CHAT_MODEL

INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
""".strip()

USER_PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()


class RAG:
    def __init__(
        self,
        index,
        llm_client: OpenAI,
        instructions: str = INSTRUCTIONS,
        prompt_template: str = USER_PROMPT_TEMPLATE,
        course: str = "llm-zoomcamp",
        model: str = CHAT_MODEL,
        num_results: int = 5,
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.course = course
        self.model = model
        self.num_results = num_results

    def search(self, question: str, course: str | None = None) -> list[dict]:
        return self.index.search(
            question,
            boost_dict={"question": 2.0, "section": 0.5},
            filter_dict={"course": course or self.course},
            num_results=self.num_results,
        )

    @staticmethod
    def build_context(search_results: list[dict]) -> str:
        lines = []
        for doc in search_results:
            lines += [doc["section"], f"Q: {doc['question']}", f"A: {doc['answer']}", ""]
        return "\n".join(lines).strip()

    def build_prompt(self, question: str, search_results: list[dict]) -> str:
        return self.prompt_template.format(
            question=question, context=self.build_context(search_results)
        )

    def llm(self, instructions: str, user_prompt: str, model: str | None = None) -> str:
        response = self.llm_client.chat.completions.create(
            model=model or self.model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    def rag(self, question: str, model: str | None = None) -> str:
        search_results = self.search(question)
        prompt = self.build_prompt(question, search_results)
        return self.llm(self.instructions, prompt, model=model)
