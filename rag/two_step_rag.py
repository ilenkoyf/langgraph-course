# Попробуем ParentInfo и MultiVector
import os

from langchain_openrouter import ChatOpenRouter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda


SYSTEM_PROMPT = """
Generate answer, based only on context. If you don't have enough information - say it.
"""

PYTHON_INFO = """
Python was conceived in the late 1980s[ by Guido van Rossum at Centrum Wiskunde & Informatica (CWI)
in the Netherlands. It was designed as a successor to the ABC programming language,
which was inspired by SETL,capable of exception handling and interfacing with the Amoeba operating system.
Python implementation began in December 1989. Van Rossum first released it in 1991 as Python 0.9.0.
Van Rossum assumed sole responsibility for the project, as the lead developer,
until 12 July 2018, when he announced his "permanent vacation" from responsibilities as Python's
"benevolent dictator for life" (BDFL); this title was bestowed on him by the Python
community to reflect his long-term commitment as the project's chief decision-maker.
(He has since come out of retirement and is self-titled "BDFL-emeritus".)
In January 2019, active Python core developers elected a five-member Steering Council to lead the project.

"""

llm = ChatOpenRouter(
    model="stepfun/step-3.5-flash:free",
    openrouter_api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    temperature=0.1,
    max_retries=2,
)
embedding_model = OpenAIEmbeddings(base_url="https://api.proxyapi.ru/openai/v1")

docs = [
    Document(PYTHON_INFO)
]

vector_store = InMemoryVectorStore(embedding_model)
vector_store.add_documents(docs)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})



def rag_step(question: str) -> str:
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"{SYSTEM_PROMPT}. Context: {context}, Question: {question}"

    return llm.invoke(prompt)


runnable = RunnableLambda(rag_step)

res = runnable.invoke("When c++ was created?")
x = 1