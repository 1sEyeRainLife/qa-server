from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

try:
    connections.get_connection("default")
except:
    connections.connect(
        alias="default",
        host="localhost",
        port="19530"
    )

fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),  # 向量，执行 page.tolist()
    FieldSchema(name="text", dtype=DataType.VARCHAR),  # 原始文本
    FieldSchema(name="doc_id", dtype=DataType.VARCHAR),
    FieldSchema(name="entity_type", dtype=DataType.VARCHAR),
    FieldSchema(name="numeric_value", dtype=DataType.FLOAT),
    FieldSchema(name="source_type", dtype=DataType.VARCHAR)
]


async def create_collection(name="entiterprise_knowledge"):
    schema = CollectionSchema(fields, description="企业行政知识库")
    collection = Collection(name, schema)
    return collection


async def create_index(collection):
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {
            "nlist": 132,
            "nprobe": 32,  # 增大nprobe提高召回率
        }
    }
    collection.create_index("embedding", index_params)

