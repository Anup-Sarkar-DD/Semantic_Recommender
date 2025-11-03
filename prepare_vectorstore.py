import pandas as pd
import numpy as np

from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

def prepare_vectorstore():
    books = pd.read_csv("books_with_emotions.csv")

    # Add large thumbnail field
    books["large_thumbnail"] = books["thumbnail"].astype(str) + "&fife=w800"
    books["large_thumbnail"] = np.where(
        books["thumbnail"].isna(),
        "cover-not-found.jpg",
        books["large_thumbnail"]
    )

    documents = []
    for idx, row in books.iterrows():
        if pd.isna(row.get("tagged_description", "")):
            continue
        doc = Document(
            page_content=row["tagged_description"],
            metadata={
                "isbn13": str(row["isbn13"]),
                "row": idx
            }
        )
        documents.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents(documents)

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db_books = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        persist_directory="books_chroma_db"
    )

    print("Vectorstore saved in 'books_chroma_db' folder")

if __name__ == "__main__":
    prepare_vectorstore()
