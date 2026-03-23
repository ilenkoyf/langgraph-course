from enum import StrEnum
import os
from pydantic import BaseModel, Field
from typing import Literal

from langchain_openrouter import ChatOpenRouter

from langgraph.graph.message import MessagesState


GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)


# StructuredOutput
class GradeDocuments(BaseModel):
    """Grade document using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes'if relevant, or 'no' if not relevant"
    )


grader_model = ChatOpenRouter(
    model="arcee-ai/trinity-mini:free",
    openrouter_api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)


def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    rewrite_count = state.get("rewrite_count", 0)

    if rewrite_count >= 1:
        return "generate_answer"

    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)

    llm_with_structured_output = grader_model.with_structured_output(GradeDocuments)
    response: GradeDocuments = llm_with_structured_output.invoke(
        [{"role": "user", "content": prompt}]
    )

    if response is None:
        return "rewrite_question"

    score = response.binary_score

    if score == "yes":
        return "generate_answer"
    return "rewrite_question"
