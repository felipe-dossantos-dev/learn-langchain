import os

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.llms.openai import OpenAI

# from langchain.chains.retrieval_qa.base import VectorDBQA
from langchain.chains import RetrievalQA
import pinecone
from langchain import hub

prompt = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")


pinecone.init(
    api_key="ccb6f354-a965-463e-9263-d23b2dd89896   ",
    environment="gcp-starter",
)

if __name__ == "__main__":
    print("Hello VectorStore!")
    loader = TextLoader("/workspaces/learn-langchain/mediumblogs/blog-1.txt")
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(len(texts))

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    docsearch = Pinecone.from_documents(
        texts, embeddings, index_name="medium-blogs-embeddings-index"
    )

    # docsearch = Pinecone.from_existing_index(
    #     "medium-blogs-embeddings-index", embeddings
    # )

    qa_chain = RetrievalQA.from_chain_type(
        OpenAI(),
        retriever=docsearch.as_retriever(),
        chain_type="stuff",
        verbose=True,
        return_source_documents=True,
    )

    # qa = RetrievalQA.from_llm(llm=OpenAI(), retriever=retriever, verbose=True)
    query = "What is a vector DB? Give me a 15 word answer for a begginner"
    result = qa_chain({"query": query})
    print(result)
