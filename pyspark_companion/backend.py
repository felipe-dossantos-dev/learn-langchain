from typing import Any
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from rich import print, pretty
pretty.install()

def run_llm(query: str) -> Any:
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    docsearch = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
        verbose=True,
    )
    return qa({"query": query})


if __name__ == "__main__":
    result = run_llm(
        query="""Can you create data for a unit test on pyspark on DataFrame with the follow schema?
        |-- age: integer (nullable = true)
 |-- workclass: string (nullable = true)
 |-- fnlwgt: integer (nullable = true)
 |-- education: string (nullable = true)
 |-- education_num: integer (nullable = true)
 |-- marital: string (nullable = true)
 |-- occupation: string (nullable = true)
 |-- relationship: string (nullable = true)
 |-- race: string (nullable = true)
 |-- sex: string (nullable = true)
 |-- capital_gain: integer (nullable = true)
 |-- capital_loss: integer (nullable = true)
 |-- hours_week: integer (nullable = true)
 |-- native_country: string (nullable = true)
 |-- label: string (nullable = true)
        """
    )
    print(result)
    print(result["result"])
