import os
import hashlib
from dotenv import load_dotenv
from pymilvus import utility, connections
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Milvus
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory

load_dotenv()


class PDFQAAgent:
    def __init__(self, pdf_path, model_name="llama3.2", persist_db=False):
        self.pdf_path = pdf_path
        self.model_name = model_name
        self.collection_name = f"pdf_{hashlib.md5(pdf_path.encode()).hexdigest()}"
        # pdf_f7a72d00deb017f28bd4954e2a05d23f

        # 0 层 模型层，最底层，与大模型交互，包括llms、chat models、prompts等组件
        # 用户问题，基于相似度计算，找到milvus中语义接近的文本，交给llm生成回答
        # 本地大模型
        self.llm = Ollama(model=self.model_name, temperature=0.7)

        # 1 层 检索层 Retrieval, 检索增强 RAG ，包括documents loaders、text splitters、vector stores、retrievers等组件
        # 文本嵌入，将pdf中的文本内容转换为高维向量（用于后续的语义搜索和相似度计算）
        self.embeddings = OllamaEmbeddings(model=self.model_name)
        if persist_db:
            try:
                connections.get_connection("default")
            except:
                connections.connect(
                    alias="default",
                    host="127.0.0.1",
                    port="19530"
                )
            if utility.has_collection(self.collection_name):
                print(f"加载已有的Milvus集合: {self.collection_name}")
                self.vectorstore = Milvus(
                    embedding_function=self.embeddings,
                    collection_name=self.collection_name,
                    connection_args={"host": "127.0.0.1", "port": "19530"}
                )
            else:
                print(f"创建新的pdf:{self.collection_name}")
                # pdf加载
                self.loader = PyPDFLoader(self.pdf_path)
                self.pages = self.loader.load()[:10]
                # 切分pdf内容
                self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
                self.texts = self.text_splitter.split_documents(self.pages)
                self.vectorstore = Milvus.from_documents(
                    documents=self.texts,
                    embedding=self.embeddings,
                    collection_name=self.collection_name,
                    connection_args={"host": "127.0.0.1", "port": "19530"}
            )
        else:
            # FAISS内存存储向量数据
            print("使用faiss内存存储")
            self.vectorstore = FAISS.from_documents(self.pages, self.embeddings)
            ...

        # 2 层 记忆层 Memory，包括短期记忆和长期记忆
        self.memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="history",
            return_messages=True,
            output_key=None
        )
        

        # 3 层，链式层,Chains
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(),
            memory=self.memory
        )
    
    def ask(self, question):
        return self.qa.run(question)
    
    def summarize(self):
        return self.qa.run("请用中文总结这篇文档的主要内容.")
    
    def extract_key_info(self):
        return self.qa.run("从文档中提取关键信息，如人名、日期、重要数据等.")
    
    def clear_memory(self):
        self.memory.clear()


if __name__ == "__main__":
    # TODO OLLAMA LLM API KEY
    # if "OPENAI_API_KEY" not in os.environ:
    #     raise ValueError("OPENAI_API_KEY必须设置")
    
    pdf_path = r"c:\Users\fanbin\Desktop\bgpconfig.pdf"

    # 创建agent
    agent = PDFQAAgent(pdf_path, persist_db=True)

    answer = agent.ask("边界网关协议是什么?")
    print(answer)

    # answer = agent.ask("请问孟凡斌是从什么时候开始工作的？")
    # print(answer)

    # answer = agent.ask("请问孟凡斌擅长哪些编程语言和框架？")
    # print(answer)