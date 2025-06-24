from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

# Chargement du CSV
df = pd.read_csv("faq_test.csv")

# Embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Emplacement de la base
db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

# Initialisation du vector store
vector_store = Chroma(
    collection_name="faq_test",
    persist_directory=db_location,
    embedding_function=embeddings,
)

# Ajout des documents s'ils n'existent pas encore
if add_documents:
    documents = []
    for i, row in df.iterrows():
        document = Document(
            page_content=f"Question: {row['Question']}\nRéponse: {row['Answer']}",
            metadata={
                "question": row["Question"],
                "answer": row["Answer"],
                "source": "faq_test.csv",
                "id": str(i)
            }
        )
        documents.append(document)
    
    vector_store.add_documents(documents=documents)
    vector_store.persist()

# Retriever pour usage dans l'app FastAPI
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1}
)