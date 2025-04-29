from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility


fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=3072),  # 向量，执行 page.tolist()
    FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=32),  # pdf或者feedback
    FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="rating", dtype=DataType.INT64),
    FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=32),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),  # 原始文本
]


schema = CollectionSchema(
    fields,
    description="企业行政知识库",
    enable_dynamic_field=True
)


async def create_index(collection):
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {
            "nlist": 132,
            "nprobe": 32,  # 增大nprobe提高召回率
        }
    }
    collection.create_index("vector", index_params)

