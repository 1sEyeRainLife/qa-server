import aioredis
import numpy as np
import random
from elasticsearch import Elasticsearch
from pymilvus import connections, Collection, utility, AnnSearchRequest, WeightedRanker, FieldSchema, CollectionSchema, DataType, RRFRanker
# from redis import Redis

# redis_conn = aioredis.create_connection("redis://localhost:6379/0")


def test_milvus():

    # 明确指定正确的地址和端口
    connections.connect(
        "default", 
        host="localhost",  # 或您的服务器IP
        port="19530"       # Milvus 默认端口
    )

    schema = CollectionSchema([
        FieldSchema("film_id", DataType.INT64, is_primary=True),
        FieldSchema("films", dtype=DataType.FLOAT_VECTOR, dim=2),
        FieldSchema("label", dtype=DataType.VARCHAR, max_length=10)
    ])
    collection = Collection("test_collection_search2", schema)
    # insert
    data = [
        [i for i in range(10)],
        [[random.random() for _ in range(2)] for _ in range(10)],
        ["auto" for _ in range(10)]
    ]
    collection.insert(data)
    index_param = {"index_type": "FLAT", "metric_type": "L2", "params": {}}
    collection.create_index("films", index_param)
    collection.create_index("label")
    collection.load()
    # search
    search_param1 = {
        "data": [[1.0, 1.0]],
        "anns_field": "films",
        "param": {"metric_type": "L2", "offset": 1},
        "limit": 2,
    }
    req1 = AnnSearchRequest(**search_param1)
    search_param2 = {
        "data": [[2.0, 2.0]],
        "anns_field": "films",
        "param": {"metric_type": "L2", "offset": 1},
        "limit": 2,
        "expr": "film_id > 0",
    }
    req2 = AnnSearchRequest(**search_param2)
    res = collection.hybrid_search([req1, req2], WeightedRanker(0.9, 0.1), 2)
    assert len(res) == 1
    hits = res[0]
    assert len(hits) == 2
    print(f"- Total hits: {len(hits)}, hits ids: {hits.ids} ")

def test_es():
    question = "hello"
    es = Elasticsearch("http://localhost:9200")
    # docs = [
    #     {"id": "1", "text": "hello world"},
    #     {"id": "2", "text": "这里是中国"},
    #     {"id": "3", "text": "hello， welcome， 中国"}
    # ]
    # for doc in docs:
    #     es.index(index="knowledge", id=doc["id"], document=doc)  # 插入一个doc
    q = {
        "query": {
            "match": {
                "text": {"query": question}  # ik_smart中文分词，基于jieba
            }
        },
        "size": 5
    }
    res = es.search(
        index="knowledge",
        body=q
    )
    print(res)
    # hits = {hit["_id"]: hit["_score"]["text"] for hit in res["hits"]["hits"]}
    # scores = np.array([hit["_score"] for hit in res["hits"]["hits"]])
    # return (hits, scores)

if __name__ == "__main__":
    test_milvus()