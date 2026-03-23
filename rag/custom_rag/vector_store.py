from langchain_openai import OpenAIEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


embedding_model = OpenAIEmbeddings(base_url="https://api.proxyapi.ru/openai/v1")
store = LocalFileStore("./cache/")
cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    embedding_model, store, namespace=embedding_model.model
)

vector_store = InMemoryVectorStore(cached_embedder)
