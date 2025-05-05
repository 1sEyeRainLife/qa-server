import os
import hashlib
import numpy as np
import jieba
import jieba.posseg as pseg
from typing import Dict
from dotenv import load_dotenv
from datetime import datetime
from pymilvus import utility, connections, Collection, DataType, WeightedRanker, RRFRanker, AnnSearchRequest
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Milvus
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from elasticsearch import Elasticsearch
from models import schema as collection_schema

load_dotenv()

# 自定义提示词模板
QA_PROMPT_TEMPLATE = """你是一位专业的技术文档分析师，请根据上下文和你的知识回答问题：
1. 如果上下文直接包含答案，引用原文回答
2. 如果上下文相关但不完整，结合知识补充回答
3. 确实无关时再说"我不知道"

上下文：{context}
关键词：{keywords}
历史记录: {history}
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
        self.memory = ConversationBufferWindowMemory()

        self.qa_prompt = PromptTemplate(
            template=QA_PROMPT_TEMPLATE,
            input_variables=["context", "keywords", "history", "question"]
        )

        self.vectorstore = None
        self.current_collection = "qa_knownledge5"
        if self.persist_db:
            self._init_milvus_connection()
            # self._load_existing_collections()
    
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
    
    def retrieval_bm25(self, question):
        """BM25关键词召回
        基于关键词的BM25相似性的搜索策略
        bm25也是es的默认搜索策略
        BM25 是一种基于统计概率的文本相似度算法，考虑了词频TF、逆文档率IDF和文档长度归一化
        提高检索精度，特定领域关键词丢失问题
        """
        # Whoosh（纯Python搜索引擎，支持BM25）
        # 分词或不分词都可以
        es = Elasticsearch("http://localhost:9200")
        # es.index(index="knowledge", id=doc[id], body={"text": doc[text]})  插入一个doc
        q = {
            "query": {
                "match": {
                    "content": question  # ik_smart中文分词，基于jieba
                }
            },
            "size": 5
        }
        res = es.search(
            index="knowledge1",
            body=q
        )
        # print("="*50)
        # print(res)
        hits = {hit["_id"]: hit["_source"]["content"] for hit in res["hits"]["hits"]}
        scores = np.array([hit["_score"] for hit in res["hits"]["hits"]])
        return (hits, scores)
    
    def retrieval_milvus(self, question):
        """向量语义找回
        基于向量相似度的查询，milvus的IVF_FLAT索引，平衡性能与准确度
        相似度衡量设置为COSIN,基于内积与长度的比值，所以与向量的长度不敏感
        如果直接使用内积比较相似性，则与向量的长度相关

        IVF_PQ索引，对向量进行乘机压缩，适用于内存敏感场景

        HNSW场景适用于千万级向量检索，召回率高达90%
        """
        question_embedding = self.embeddings.embed_query(question)
        keywords = self._extrace_keywords(question)
        collection = Collection(self.current_collection)
        collection.load()
        print(f"集合行数: {collection.num_entities}")
        print(f"集合字段: {collection.schema}")
        _res = collection.query(expr='source_type == "pdf"', output_fields=["content"], limit=1)
        print(f"插入的原始数据示例: {_res}")
        expr = f"text like '{keywords[0].lower().strip(".,!?")}%'" if keywords else None
        print(f"关键字： {expr}")
        # results1 = collection.search(
        #     [question_embedding],
        #     "vector",
        #     {
        #         "metric_type": "L2",
        #         "params": {"nprobe": 16},
        #     },
        #     5,
        #     expr=expr,  # 这里是标签过滤
        #     output_fields=["text", "source_type", "user_id"]
        # )
        # results1 = AnnSearchRequest(**{
        #     "data": [question_embedding],
        #     "anns_field": "vector",
        #     "param": {"metric_type": "L2", "params": {"nprobe": 16}},
        #     "limit": 5,
        #     "expr": expr
        # })
        results = collection.search(
            [question_embedding],
            "vector",
            {
                "metric_type": "L2",
                "params": {"nprobe": 16},  # 平衡精度与速度
            },
            5,
            output_fields=["text", "source_type", "user_id"]
        )
        # results2 = AnnSearchRequest(**{
        #     "data": [question_embedding],
        #     "anns_field": "vector",
        #     "param": {"metric_type": "L2", "params": {"nprobe": 16}},
        #     "limit": 5
        # })
        # collections. 融合重排序  
        # 0.2, 0.5
        # reqs = [results1, results2]
        # ranker = WeightedRanker(0.2, 0.5)
        # hybrid_res = collection.hybrid_search(reqs, ranker, limit=5, output_fields=["text", "source_type", "user_id"])
        hits = {hit.id: hit.entity.get("text") for hit in results[0]}
        scores = np.array([1/hit.distance for hit in results[0]])  # 转换距离为相似度
        # ES的BM25分数和milvus的距离分数需要反向处理，距离越小越相关
        return (hits, scores)
    
    def normalize_scores(self, scores):
        """Min-Max标准化到[0,1]范围"""
        min_val = np.min(scores)
        max_val = np.max(scores)
        return (scores-min_val) / (max_val - min_val + 1e-6)  # 避免除0
    
    def hybrid_score(self, bm25_scores, vector_scores, bm25_weight=0.3):
        """线性加权融合"""
        combined = bm25_weight * bm25_scores + (1-bm25_weight) * vector_scores
        return combined
    
    def rrf_hybrid(self, bm25_ranks, vector_ranks, k=60):
        """RRF 倒数排序融合"""
        scores = {}
        for doc_id in set(bm25_ranks + vector_ranks):
            rank_bm25 = bm25_ranks.index(doc_id) + 1 if doc_id in bm25_ranks else k
            rank_vector = vector_ranks.index(doc_id) + 1 if doc_id in vector_ranks else k
            scores[doc_id] = 1/(rank_bm25+k) + 1/(rank_vector+k)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    def hybrid_rank(self, bm25_res, vector_res):
        """基于权重的融合排序"""
        # print("-"*100)
        # print(vector_res)
        bm25_hits, bm25_scores = bm25_res
        vector_hits, vector_scores = vector_res
        all_doc_ids = list(set(bm25_hits.keys()).union(set(vector_hits.keys())))

        # 分数标准化与填充
        def fill_scores(hits, all_ids, original_scores):
            scores = np.zeros(len(all_ids))
            for i, doc_id in enumerate(all_ids):
                if doc_id in hits:
                    idx = list(hits.keys()).index(doc_id)
                    # print(">"*50, original_scores)
                    # print(">"*50, idx)
                    scores[i] = original_scores[idx]
            return self.normalize_scores(scores)
    
        bm25_norm = fill_scores(bm25_hits, all_doc_ids, bm25_scores)
        vector_norm = fill_scores(vector_hits, all_doc_ids, vector_scores)
        # 加权融合
        combined_scores = self.hybrid_score(bm25_norm, vector_norm, 0.3)
        ranked_indices = np.argsort(combined_scores)[::-1]  # 倒叙
        results = []
        for idx in ranked_indices:
            doc_id = all_doc_ids[idx]
            source = "BM25" if doc_id in bm25_hits else "Vector"
            if doc_id in bm25_hits and doc_id in vector_hits:
                source = "Both"
            results.append({
                "rank": len(results)+1,
                "doc_id": doc_id,
                "source": source,
                "score": combined_scores[idx],
                "text": bm25_hits.get(doc_id) or vector_hits.get(doc_id)
            })
        return results
    
    def retrieval_hybrid(self, question):
        bm25_docs, _ = self.retrieval_bm25(question)
        vector_docs, _ = self.retrieval_milvus(question)
        bm25_docs.update(vector_docs)
        return bm25_docs, "test"

    def ask(self, question):
        bm25_res = self.retrieval_bm25(question)
        vector_res = self.retrieval_milvus(question)

        # 融合排序
        res = self.hybrid_rank(bm25_res, vector_res)
        # bm25_res[0] = {hit.id: hit.entity.get("text") for hit in results[0]}

        # 向量
        # res = [{"text": v} for k, v in bm25_res[0].items()]
        
        context = "\n\n".join([r['text'] for r in res])
        print("上下文: ", res)
        # return self.llm(f"根据以下信息回答问题：\n{context}\n\n问题：{question}")
        
        # 动态调整提示词：
        if "是什么" in question:
            question = f"请用通俗易懂的语言解释：{question}"
        elif "如何" in question:
            question = f"请分步骤说明：{question}"
        
        keywords = self._extrace_keywords(question)
        history = self.memory.load_memory_variables({})
    
        answer = self.llm(
            self.qa_prompt.format(
                context=context,
                keywords=keywords,
                history=history.get("history", ""),
                question=f"问题: {question}\n请根据上下文用中文回答;")
        )
        # answer = self.qa.run(
        #     context=context,
        #     keywords=",".join(keywords),
        #     question=question
        # )
        # answer = self.qa.run(question)
        return answer
    
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