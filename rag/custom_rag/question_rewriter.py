from langchain.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph.message import MessagesState


REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)


class QuestionRewriter:
    def __init__(self, llm: BaseChatModel):
        self._llm = llm

    def rewrite_question(self, state: MessagesState) -> dict:
        "Rewrite the original user question."
        messages = state["messages"]
        question = next(
            (m.content for m in reversed(messages) if m.type == "human"),
            messages[0].content,
        )
        rewrite_count = state.get("rewrite_count", 0)

        prompt = REWRITE_PROMPT.format(question=question)

        response = self._llm.invoke([{"role": "user", "content": prompt}])

        return {
            "messages": [HumanMessage(content=response.content)],
            "rewrite_count": rewrite_count + 1,
        }
