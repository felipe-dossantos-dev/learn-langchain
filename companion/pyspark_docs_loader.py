from langchain.document_loaders import JSONLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["title"] = record.get("title")
    metadata["url"] = record.get("url")
    metadata["lib"] = "pyspark"
    return metadata


# load the document and split it into chunks
loader = JSONLoader(
    file_path="./companion/pyspark_docs.jsonl",
    content_key="content",
    jq_schema=".",
    text_content=True,
    json_lines=True,
    metadata_func=metadata_func,
)

documents = loader.load()
documents = [
    d for d in documents if d.metadata["title"] is not None and not d.page_content.isspace()
]

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# load it into Chroma
db = Chroma.from_documents(docs, embedding_function, persist_directory="./chroma_db")

# load from disk
# db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

# query it
query = "How to read a csv file to a DataFrameReader?"
docs = db.similarity_search(query, k=3)
for d in docs:
    print("")
    print("-------------")
    print(d.page_content)


