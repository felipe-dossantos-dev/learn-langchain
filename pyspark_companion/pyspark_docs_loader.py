from typing import Any
from langchain.document_loaders import JSONLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["title"] = record.get("title")
    metadata["url"] = record.get("url")
    return metadata


# load the document and split it into chunks
# loader = JSONLoader(
#     file_path="./scrapers/docs.jsonl",
#     content_key="content",
#     jq_schema=".",
#     text_content=True,
#     json_lines=True,
#     metadata_func=metadata_func,
# )

# documents = loader.load()
# documents = [
#     d for d in documents if d.metadata["title"] is not None and not d.page_content.isspace()
# ]

# split it into chunks
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)

# create the open-source embedding function
# embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# load it into Chroma
# db = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")

# load from disk
# db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

# query it
# query = "How to read a csv file to a DataFrameReader?"
# docs = db.similarity_search(query, k=3)
# for d in docs:
#     print("")
#     print("-------------")
#     print(d.page_content)


def run_llm(query: str) -> Any:
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    docsearch = Chroma(
        persist_directory="./chroma_db", embedding_function=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
        verbose=True
    )
    return qa({"query": query})


if __name__ == "__main__":
    print(run_llm(query="How to read a csv file from spark?"))
