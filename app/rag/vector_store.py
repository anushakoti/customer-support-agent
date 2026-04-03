from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

faq_data = [
    "You can return items within 30 days",
    "Shipping takes 3-5 business days",
    "Refunds processed within 5 days"
]

embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_texts(faq_data, embeddings)

def search(query: str):
    docs = vector_store.similarity_search(query, k=1)
    return docs[0].page_content if docs else "No relevant info"