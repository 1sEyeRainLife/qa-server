import os
import hashlib
from typing import Dict
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
    def __init__(self, model_name="llama3.2", persist_db=False):
        self.model_name = model_name
        self.persist_db = persist_db
        # 
        # pdf_f7a72d00deb017f28bd4954e2a05d23f

        # 0 层 模型层，最底层，与大模型交互，包括llms、chat models、prompts等组件
        # 用户问题，基于相似度计算，找到milvus中语义接近的文本，交给llm生成回答
        # 本地大模型
        self.llm = Ollama(model=self.model_name, temperature=0.7)

        self.embeddings = OllamaEmbeddings(model=self.model_name)
        # 2 层 记忆层 Memory，包括短期记忆和长期记忆
        self.memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="history",
            return_messages=True,
            output_key=None
        )

        self.vectorstore = None
        self.qa = None
        self.qa_chains: Dict[str, RetrievalQA] = {}
        self.current_collection = None
        if self.persist_db:
            self._init_milvus_connection()
            self._load_existing_collections()
    
    def _init_milvus_connection(self):
        try:
            connections.get_connection("default")
        except:
            connections.connect(
                alias="default",
                host="127.0.0.1",
                port="19530"
            )
    
    def _load_existing_collections(self):
        collections = utility.list_collections()
        pdf_collections = [col for col in collections if col.startswith("pdf_")]
        for col_name in pdf_collections:
            print(f"加载已有集合: {col_name}")
            vectorstore = Milvus(
                embedding_function=self.embeddings,
                collection_name=col_name,
                connection_args={"host": "127.0.0.1", "port": "19530"}
            )
            self.qa_chains[col_name] = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(),
                memory=ConversationBufferWindowMemory(
                    k=5,
                    memory_key="history",
                    return_messages=True,
                    output_key=None
                )
            )
            self.current_collection = col_name

    def load_pdf(self, pdf_path):
        # 1 层 检索层 Retrieval, 检索增强 RAG ，包括documents loaders、text splitters、vector stores、retrievers等组件
        # 文本嵌入，将pdf中的文本内容转换为高维向量（用于后续的语义搜索和相似度计算）
        self.current_collection = f"pdf_{hashlib.md5(pdf_path.encode()).hexdigest()}"
        if self.current_collection in self.qa_chains:
            print(f"集合{self.current_collection}已存在。。。")
            return
        
        # pdf加载
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()[:10]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        texts = text_splitter.split_documents(pages)
        vectorstore = Milvus.from_documents(
            documents=texts,
            embedding=self.embeddings,
            collection_name=self.current_collection,
            connection_args={"host": "127.0.0.1", "port": "19530"}
        )
        self.qa_chains[self.current_collection] = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            memory=ConversationBufferWindowMemory(
                    k=5,
                    memory_key="history",
                    return_messages=True,
                    output_key=None
                )
        )
            # FAISS内存存储向量数据
            # print("使用faiss内存存储")
            # self.vectorstore = FAISS.from_documents(self.pages, self.embeddings)
            # ...

        # 3 层，链式层,Chains
    
    def ask(self, question, collection_name=None):
        target_collection = collection_name or self.current_collection
        return self.qa_chains[target_collection].run(question)
    
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