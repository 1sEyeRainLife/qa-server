import os
import hashlib
import numpy as np
import jieba
import jieba.posseg as pseg
from typing import Dict
from dotenv import load_dotenv
from datetime import datetime
from pymilvus import utility, connections, Collection, DataType
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Milvus
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from models import schema as collection_schema

load_dotenv()

# 自定义提示词模板
QA_PROMPT_TEMPLATE = """你是一位专业的技术文档分析师，请根据上下文和你的知识回答问题：
1. 如果上下文直接包含答案，引用原文回答
2. 如果上下文相关但不完整，结合知识补充回答
3. 确实无关时再说"我不知道"

上下文：{context}
关键词：{keywords}
问题：{question}
答案（简洁中文）：
"""

# 关键词：{keywords}  # 显式提示LLM关注这些词
# 提示词优化技巧
# 在提示词中指定AI的角色，例如： 你是以为技术文档专家，请回答与i下问题。。。
# 明确输出的格式，如列表，json等
# 负面提示，排除不想要的答案，例如：不要列举无关的示例


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

        self.embeddings = OllamaEmbeddings(model="quentinz/bge-base-zh-v1.5")
        # 2 层 记忆层 Memory，包括短期记忆和长期记忆
        self.memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="history",
            return_messages=True
        )

        self.qa_prompt = PromptTemplate(
            template=QA_PROMPT_TEMPLATE,
            input_variables=["context", "keywords", "question"]
        )

        self.vectorstore = None
        self.qa = None
        self.qa_chains: Dict[str, RetrievalQA] = {}
        self.current_collection = "qa_knownledge5"
        if self.persist_db:
            self._init_milvus_connection()
            self._load_existing_collections()
    
    def _extrace_keywords(self, text):
        words = pseg.cut(text)
        return [word for word, flag in words if flag in ['n', 'nr', 'ns']]  # 提取名词
    
    def _init_milvus_connection(self):
        try:
            connections.get_connection("default")
        except:
            connections.connect(
                alias="default",
                host="127.0.0.1",
                port="19530"
            )
        if not utility.has_collection(self.current_collection):
            # 创建集合
            collection = Collection(name=self.current_collection, schema=collection_schema)
            # 在正确的字段上创建索引
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "L2",
                "params": {"nlist": 128}
            }
            collection.create_index(field_name="vector", index_params=index_params)  # 使用实际字段名
            print("索引创建成功")
    
    def _load_existing_collections(self):
        if utility.has_collection(self.current_collection) and self.persist_db:
            print(f"加载已有集合: {self.current_collection}")
            self.vectorstore = Milvus(
                embedding_function=self.embeddings,
                collection_name=self.current_collection,
                connection_args={"host": "127.0.0.1", "port": "19530"}
            )
            self.qa = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(),
                input_key="question",
                # memory=self.memory,
                chain_type_kwargs={"prompt": self.qa_prompt}
            )
        # collections = utility.list_collections()
        # pdf_collections = [col for col in collections if col.startswith("pdf_")]
        # for col_name in pdf_collections:
        #     print(f"加载已有集合: {col_name}")
        #     vectorstore = Milvus(
        #         embedding_function=self.embeddings,
        #         collection_name=col_name,
        #         connection_args={"host": "127.0.0.1", "port": "19530"}
        #     )
        #     self.qa_chains[col_name] = RetrievalQA.from_chain_type(
        #         llm=self.llm,
        #         chain_type="stuff",
        #         retriever=vectorstore.as_retriever(),
        #         memory=ConversationBufferWindowMemory(
        #             k=5,
        #             memory_key="history",
        #             return_messages=True,
        #             output_key=None
        #         ),
        #         chain_type_kwargs={"prompt": self.qa_prompt}
        #     )
        #     self.current_collection = col_name
    
    def load_pdf(self, pdf_path, merge_to_existing=True):
        print("准备加载pdf")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        texts = text_splitter.split_documents(pages)

        print(f"切分完成...共计{len(texts)}")
        collection = Collection(self.current_collection)
        collection.load()

        batch_size = 50
        print(f"开始分批写入。。。每批100，需{len(texts)// batch_size}轮")
        for batch_start in range(0, len(texts), batch_size):
            batch_end = min(batch_start+batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]
            # 准备数据
            data = {
                "ids": [],
                "embeddings": [],
                "sources": [],
                "user_ids": [],
                "ratings": [],
                "timestamps": [],
                "contents": []
            }
            for i, text in enumerate(batch_texts):
                data["ids"].append(f"pdf_{hashlib.md5(pdf_path.encode()).hexdigest()}_{i}")
                data["embeddings"].append(self.embeddings.embed_query(text.page_content))
                data["sources"].append("pdf")
                data["user_ids"].append("")
                data["ratings"].append(0)
                data["timestamps"].append("")
                data["contents"].append(text.page_content)
            # 验证
            n_rows = len(data["ids"])
            assert all(len(lst) == n_rows for lst in data.values()), "字段行数不一致！"
            print(f"准备插入 {n_rows} 行数据，向量维度: {len(data['embeddings'][0])}")

            schema = collection.schema
            for field in schema.fields:
                if field.dtype == DataType.FLOAT_VECTOR:
                    print(f"向量字段 '{field.name}' 的维度: {field.params['dim']}")
            
            # 按Schema字段顺序组织
            insert_data = [
                data["ids"],
                data["embeddings"],  # 注意这里是二维列表
                data["sources"],
                data["user_ids"],
                data["ratings"],
                data["timestamps"],
                data["contents"]
            ]
            collection.insert(insert_data)
        collection.flush()
        print(f"共插入: {len(texts)} 数据")

    def load_pdf2(self, pdf_path, merge_to_existing=True):
        # 1 层 检索层 Retrieval, 检索增强 RAG ，包括documents loaders、text splitters、vector stores、retrievers等组件
        # 文本嵌入，将pdf中的文本内容转换为高维向量（用于后续的语义搜索和相似度计算）
        # self.current_collection = f"pdf_{hashlib.md5(pdf_path.encode()).hexdigest()}"
        # if self.current_collection in self.qa_chains:
        #     print(f"集合{self.current_collection}已存在。。。")
        #     return
        
        # pdf加载
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()[:10]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        texts = text_splitter.split_documents(pages)
        ids = [hashlib.md5(f"{pdf_path}_{i}".encode()).hexdigest() for i in range(len(texts))]
        # 添加来源标记，文件名
        for t in texts:
            t.metadata["source"] = os.path.basename(pdf_path)
        if merge_to_existing and self.current_collection and self.qa:
            print("合并到已存在的集合")
            existing_store = Milvus(
                embedding_function=self.embeddings,
                collection_name=self.current_collection,
                connection_args={"host": "127.0.0.1", "port": "19530"}
            )
            existing_store.add_documents(texts, ids=ids)
        else:
            print("创建新的集合")
            vectorstore = Milvus.from_documents(
                documents=texts,
                embedding=self.embeddings,
                collection_name=self.current_collection,
                connection_args={"host": "127.0.0.1", "port": "19530"},
                ids=ids
            )
            self.qa = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(),
                memory=self.memory,
                chain_type_kwargs={"prompt": self.qa_prompt}
            )
            # FAISS内存存储向量数据
            # print("使用faiss内存存储")
            # self.vectorstore = FAISS.from_documents(self.pages, self.embeddings)
            # ...

        # 3 层，链式层,Chains

    def add_feedback(self, feedback: dict):
        # 生成反馈内容的向量
        embedding = self.embeddings.embed_query(feedback["content"])

        # 准备插入的结构化数据
        data = [
            [feedback["feedback_id"]],
            [embedding],
            ["feedback"],
            [feedback.get("user_id", "anonymous")],
            [feedback.get("rating", 0)],
            [feedback.get("timestamp", datetime.now().isoformat())],
            [feedback["content"]]
        ]
        collection = Collection(self.current_collection)
        collection.insert(data)
        print("反馈已添加")
    
    def ask(self, question, collection_name=None):
        target_collection = collection_name or self.current_collection
        # 动态调整提示词：
        if "是什么" in question:
            question = f"请用通俗易懂的语言解释：{question}"
        elif "如何" in question:
            question = f"请分步骤说明：{question}"
        question_embedding = self.embeddings.embed_query(question)
        keywords = self._extrace_keywords(question)
        # conditions = []
        # if keywords:
        #     for kw in keywords:
        #         conditions.append(f"text like '{kw.lower().strip(".,!?")}%'")
        #         break
        # expr = " OR ".join(conditions) if conditions else None
        expr = f"text like '{keywords[0].lower().strip(".,!?")}%'" if keywords else None
        # results = self.vectorstore.search(
        #     question_embedding,
        #     "similarity",
        #     # output_fields=["content", "source_type", "user_id"]
        # )
        collection = Collection(self.current_collection)
        collection.load()
        print(f"集合行数: {collection.num_entities}")
        print(f"集合字段: {collection.schema}")
        _res = collection.query(expr='source_type == "pdf"', output_fields=["content"], limit=1)
        print(f"插入的原始数据示例: {_res}")
        print(f"关键字： {expr}")
        results1 = collection.search(
            [question_embedding],
            "vector",
            {
                "metric_type": "L2",
                "params": {"nprobe": 16},
            },
            5,
            expr=expr,
            output_fields=["text", "source_type", "user_id"]
        )
        results2 = collection.search(
            [question_embedding],
            "vector",
            {
                "metric_type": "L2",
                "params": {"nprobe": 16},
            },
            5,
            output_fields=["text", "source_type", "user_id"]
        )
        # return self.qa.run(question)
        # 组装上下文供LLM生成答案
        # context = "\n\n".join([
        #     f"[来源: {hit.entity.get('source_type')}, 用户: {hit.entity.get('user_id')}]\n{hit.entity.get('content')}"
        #     for hit in results
        # ])
        res = []
        for hits in results1:
            for hit in hits:
                entity = hit.entity
                res.append(
                    f"[来源: {entity.get('source_type')}, 用户: {entity.get('user_id')}]\n{entity.get('text')}"
                )
        for hits in results2:
            for hit in hits:
                entity = hit.entity
                res.append(
                    f"[来源: {entity.get('source_type')}, 用户: {entity.get('user_id')}]\n{entity.get('text')}"
                )
        context = "\n\n".join(res)
        print("上下文: ", res)
        # return self.llm(f"根据以下信息回答问题：\n{context}\n\n问题：{question}")
        answer = self.llm(
            self.qa_prompt.format(context=context, keywords=keywords, question=f"问题: {question}\n请根据上下文用中文回答;")
        )
        # answer = self.qa.run(
        #     context=context,
        #     keywords=",".join(keywords),
        #     question=question
        # )
        # answer = self.qa.run(question)
        return answer
    
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