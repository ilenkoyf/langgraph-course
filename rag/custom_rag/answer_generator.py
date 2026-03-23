from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.message import MessagesState


GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)


class AnswerGenerator:
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def generate_answer(self, state: MessagesState) -> dict:
        """Generate an answer."""
        messages = state["messages"]
        question = next(
            (m.content for m in reversed(messages) if m.type == "human"),
            messages[0].content,
        )
        context = messages[-1].content

        prompt = GENERATE_PROMPT.format(question=question, context=context)

        response = self._llm.invoke([{"role": "user", "content": prompt}])
        return {"messages": [response]}
