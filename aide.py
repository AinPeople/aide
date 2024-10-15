from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# List of URLs to load documents from
urls = [
    "https://www.assembly.go.kr/members/22nd/LEEJAEMYUNG",
    "https://ko.wikipedia.org/wiki/%EC%9D%B4%EC%9E%AC%EB%AA%85",
    "https://blog.naver.com/PostView.naver?blogId=jaemyunglee&logNo=223584109144&categoryNo=54&parentCategoryNo=54&from=thumbnailList",
]
# Load documents from the URLs
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]


# Initialize a text splitter with specified chunk size and overlap
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
# Split the documents into chunks
doc_splits = text_splitter.split_documents(docs_list)


from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
# Create embeddings for documents and store them in a vector store
vectorstore = SKLearnVectorStore.from_documents(
    documents=doc_splits,
    embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
)
retriever = vectorstore.as_retriever(k=4)


from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
# Define the prompt template for the LLM
prompt = PromptTemplate(
    template="""대한민국 국회의원실 막내 보좌관으로 답변을 시작하십시오.
    You are an assistant for question-answering tasks.
    Use the following documents to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.
    Always respond in Korean.:
    Question: {question}
    Documents: {documents}
    Answer:
    """,
    input_variables=["question", "documents"],
)


# Initialize the LLM with Llama 3.1 model
llm = ChatOllama(
    model="llama3.1",
    temperature=0,
)

# print("==llm.eos_token_id==", llm.eos_token_id)
# print("==llm.pad_token_id==", llm.pad_token_id)


# Create a chain combining the prompt template and LLM
rag_chain = prompt | llm | StrOutputParser()


# Define the RAG application class
class RAGApplication:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain
    def run(self, question):
        # Retrieve relevant documents
        documents = self.retriever.invoke(question)
        #print("======documents:", documents)
        # Extract content from retrieved documents
        doc_texts = "\\n".join([doc.page_content for doc in documents])
        #print("======doc_texts:", doc_texts)
        # Get the answer from the language model
        answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
        return answer


# Initialize the RAG application
rag_application = RAGApplication(retriever, rag_chain)
# Example usage
#question = "이재명 의원에 대해서 알려 주세요."
question = "이재명은 어떤 사람입니까?"

print("질문시작")
answer = rag_application.run(question)
print("답변시작")
print("-----Question:", question)
print("-----Answer:", answer)