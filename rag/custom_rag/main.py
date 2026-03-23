import operator
from typing import Annotated, TypedDict
from uuid import uuid4
import os

from langchain_openrouter import ChatOpenRouter
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph.message import MessagesState, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, START, END

from custom_rag.loader import WebLoader
from custom_rag.vector_store import vector_store
from custom_rag.document_grader import grade_documents
from custom_rag.question_rewriter import QuestionRewriter
from custom_rag.answer_generator import AnswerGenerator


urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

loader = WebLoader(urls)
docs = loader.get_docs()


text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100,
    chunk_overlap=5,
)
splits = text_splitter.split_documents(docs)


vector_store.add_documents(splits)
retriever = vector_store.as_retriever()


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    rewrite_count: Annotated[int, operator.or_]


@tool
def retrieve_information(query: str) -> str:
    "Search and return information from vector store"
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])


llm = ChatOpenRouter(
    model="stepfun/step-3.5-flash:free",
    openrouter_api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0.1,
    max_retries=1,
)
llm_with_tools = llm.bind_tools([retrieve_information])

question_rewriter = QuestionRewriter(llm)
answer_generator = AnswerGenerator(llm)


def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool,
    or simply respond to the user.
    """
    response = llm_with_tools.invoke(state["messages"])

    return {"messages": [response]}


config = {"configurable": {"thread_id": str(uuid4)}}
workflow = StateGraph(State)

workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retrieve_information]))
workflow.add_node(question_rewriter.rewrite_question)
workflow.add_node(answer_generator.generate_answer)


workflow.add_edge(START, "generate_query_or_respond")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "generate_query_or_respond", tools_condition, {"tools": "retrieve", END: END}
)

# After retrieve edges
workflow.add_conditional_edges(
    "retrieve",
    grade_documents,
)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

graph = workflow.compile()


# for chunk in graph.stream(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "What does Lilian Weng say about types of reward hacking?",
#             }
#         ]
#     }
# ):
#     for node, update in chunk.items():
#         print("Update from node", node)
#         update["messages"][-1].pretty_print()
#         print("\n\n")
