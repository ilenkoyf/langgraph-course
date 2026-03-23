import time

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.vectorstores import InMemoryVectorStore


docs = [
    Document("Python is the best programming language", metadata={"source": "insta"}),
    Document("Python is dynamic typing language", metadata={"source": "python.org"}),
    Document("C++ is the worst programming language", metadata={"source": "insta"}),
    Document("Go is the medium programming language", metadata={"source": "insta"})
]

embeddings_model = OpenAIEmbeddings(base_url="https://api.proxyapi.ru/openai/v1")

store = LocalFileStore("./cache/")

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    embeddings_model,
    store,
    namespace=embeddings_model.model
)

vector_store = InMemoryVectorStore(cached_embedder)
vector_store.add_documents(docs)


def filter_by_insta(doc: Document) -> Document | None:
    if doc.metadata["source"] == "insta":
        return doc

res = vector_store.similarity_search("Python typing", k=1, filter=filter_by_insta)

x = 1

