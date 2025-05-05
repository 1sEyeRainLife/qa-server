import json
import random
from elasticsearch import Elasticsearch
from pymilvus import connections, Collection
from faker import Faker
from data import test_questions


def set_mappings(es: Elasticsearch, name: str):
    res = es.indices.exists(index=name)
    if res:
        return
    settings = {
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "id": { "type": "keyword" },
                "content": { "type": "text" },
                "source_type": { "type": "keyword" },
                "department_level": { "type": "integer" }
            }
        }
    }
    es.indices.create(index=name, body=settings)

def insert_test_qa(es: Elasticsearch, name: str):
    from agent import PDFQAAgent
    rag_system = PDFQAAgent(persist_db=True)
    fake = Faker()
    for tq in test_questions:
        for doc_id in tq["relevant_docs"]:
            pre = fake.sentence(nb_words=random.randint(10, 100))
            tail = fake.sentence(nb_words=random.randint(1, 20))
            text = f"{pre}\n{tq['question']}{tq['answer']}\n{tail}"
            es.index(
                index=name,
                id=doc_id,
                body={"id": doc_id, "content": text, "source_type": "manual", "department_level": 2}
            )
            rag_system.add_feedback({
                "feedback_id": doc_id,
                "content": text
            })

def load_documents(es: Elasticsearch, index_name: str):
    collection = Collection("qa_knownledge5")
    collection.load()
    limit = collection.num_entities
    print(f"milvus中共计{limit}记录")
    # docs = collection.query(expr="source_type == 'feedback' || source_type == 'pdf'", limit=limit, output_fields=["id", "text"])
    docs = collection.query(expr="", output_fields=["id", "text"], limit=limit)
    for doc in docs:
        id_ = doc["id"]
        es.index(index=index_name, id=id_, body={"id": id_, "content": doc["text"], "source_type": "milvus", "department_level": 1})
    collection.release()

def search(es: Elasticsearch, index_name: str, q: str):
    count_res = es.count(index=index_name)
    print(f"es中共计{count_res['count']}记录")
    res = es.search(
        index=index_name,
        body={
            "query": {
                "match": {
                    "content": {"query": q}
                }
            },
            "size": 10
        }
    )
    return res

if __name__ == "__main__":
    # connections.connect(
    #         "default", 
    #         host="localhost",  # 或您的服务器IP
    #         port="19530"       # Milvus 默认端口
    #     )
    # # collection = Collection("qa_knownledge5")
    # # collection.load()
    # # res = collection.query(expr="", limit=1)
    # # print(res[0])
    

    name = "knowledge1"
    es = Elasticsearch("http://localhost:9200")
    # # set_mappings(es, name)

    # # load_documents(es, name)
    # # insert_docs(es, name, docs)

    # res = search(es, name, "现代化的企业建设时怎么样的？")
    # for hit in res["hits"]["hits"]:
    #     print(hit["_score"], hit["_source"]["content"][:50])
    # # with open(r"C:\Users\fanbin\Desktop\auth\search.json", "w") as f:
    # #     json.dump(data, f)


    insert_test_qa(es, name)