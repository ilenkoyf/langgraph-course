
from bs4.filter import SoupStrainer

from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader


class WebLoader:
    def __init__(self, urls: str):
        self._urls = urls
    
    @property
    def loader(self):
        if not hasattr(self, "_loader"):
            bs4_strainer = SoupStrainer(class_=("post-title", "post-header", "post-content"))
            self._loader = WebBaseLoader(
                web_paths=self._urls,
                bs_kwargs={"parse_only": bs4_strainer},
            )
        return self._loader
    
    def get_docs(self) -> list[Document]:
        return self.loader.load()
