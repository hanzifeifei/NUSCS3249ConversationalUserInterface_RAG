"""Vector RAG with LangChain: answer from the clinic's documents.

The same pipeline as Lecture 7 (LangChain RAG tutorial), with real LangChain parts:

  STAGE 1 · INDEXING (once, when the app starts)
      load_docs()  ->  split_docs()  ->  build_vector_store()
      Load           Split            Embed + Store

  STAGE 2 · RETRIEVAL (every message)
      vector_search(question)  ->  the k most similar chunks, with scores

LangChain parts used here:
  Document                         langchain_core.documents
  MarkdownHeaderTextSplitter,      langchain_text_splitters
  RecursiveCharacterTextSplitter
  Embeddings (interface)           langchain_core.embeddings
  InMemoryVectorStore              langchain_core.vectorstores
"""
import math
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
CHUNK_SIZE = 500        # characters
CHUNK_OVERLAP = 50
TOP_K = 3
MIN_SCORE = 0.15        # cosine similarity; ignore weaker matches


# ==========================================================================
# Given: loading, and an embedding model that runs offline
# ==========================================================================
def load_docs(folder=DOCS_DIR):
    """1 · LOAD: every .md file in docs/ as a LangChain Document, with its file name as source."""
    return [Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": p.name})
            for p in sorted(Path(folder).glob("*.md"))]


STOPWORDS = set("""a an and are as at be been but by can do does for from has have how i if in
into is it its me my not of on or our so that the their them there these they this to up
was we what when where which who will with you your yours""".split())


def tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
            for w in words if w not in STOPWORDS]


class KeywordEmbeddings(Embeddings):
    """A LangChain embedding model that needs no download and no key: TF-IDF.

    fit() learns the vocabulary from the chunks; then any text becomes a
    vector with one number per word. A word counts more if it is rare across
    the chunks. It matches WORDS, not meaning ("mum" is not "family").

    To use a neural model instead, change ONE line in build_vector_store(), e.g.
        OpenAIEmbeddings(model="text-embedding-3-small")   # pip install langchain-openai
    Everything else stays the same: that is what the Embeddings interface is for.
    """

    def __init__(self, vocab=None, idf=None):
        self.vocab = vocab or {}
        self.idf = idf or []

    @classmethod
    def fit(cls, texts):
        docs = [set(tokenize(t)) for t in texts]
        words = sorted(set().union(*docs)) if docs else []
        n = len(docs)
        idf = [math.log((1 + n) / (1 + sum(w in d for d in docs))) + 1 for w in words]
        return cls({w: i for i, w in enumerate(words)}, idf)

    def _embed(self, text):
        vector = [0.0] * len(self.vocab) + [1e-6]    # last number: never an all-zero vector
        for word in tokenize(text):
            if word in self.vocab:
                i = self.vocab[word]
                vector[i] += self.idf[i]
        return vector

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


# ==========================================================================
# Activity 1: vector RAG
# ==========================================================================
def split_docs(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """1.1  SPLIT: Markdown sections first, then long sections into smaller pieces.

    - by_heading = MarkdownHeaderTextSplitter(
          headers_to_split_on=[("#", "title"), ("##", "section")], strip_headers=False)
    - by_size = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    - for each doc: sections = by_heading.split_text(doc.page_content)
                    pieces = by_size.split_documents(sections)
      and give each piece metadata source=doc's source and id=f"{source}#{n}" (n from 1)
    - return all the pieces (a list of Documents)
    """
    return []                                            # TODO


def build_vector_store(chunks, embeddings=None):
    """1.2  EMBED + STORE.

    - embeddings = embeddings or KeywordEmbeddings.fit([c.page_content for c in chunks])
    - store = InMemoryVectorStore(embedding=embeddings); store.add_documents(chunks)
    - return store
    """
    return None                                          # TODO


def vector_search(query, store, k=TOP_K, min_score=MIN_SCORE):
    """1.3  RETRIEVE: the k most similar chunks, best first, each scoring at least min_score.

    - results = store.similarity_search_with_score(query, k=k)   -> [(Document, score), ...]
    - keep the pairs with score >= min_score, as
      {"id": d.metadata["id"], "source": d.metadata["source"], "text": d.page_content, "score": round(score, 3)}
    """
    return []                                            # TODO


# ==========================================================================
# Given: turning passages into text for the LLM
# ==========================================================================
def format_passages(passages):
    """One line per passage, tagged with its id: [policies.md#3] ..."""
    if not passages:
        return "none matched"
    return "\n".join(f"[{p['id']}] {' '.join(p['text'].split())}" for p in passages)
