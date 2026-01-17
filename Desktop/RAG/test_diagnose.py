from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader

print("Starting execution...")

doc = Document(
    page_content="MAIN TEXT CONTENT USED FOR THIS RAG",
    metadata={
        "source": "exmaple.txt",
        "pages": [1, 2, 3],
        "author": "Prajwal",
        "date_created": "2025-12-31"
    }
)
print("Document created:", doc)

print("Imports successful.")
