
from numpy import size
from pymilvus import Collection, DataType

from pymilvus import FieldSchema, CollectionSchema

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from pymilvus.exceptions import ConnectionNotExistException

try:
    # Connect to Milvus server
    connections.connect(alias='default', host='localhost', port='19530', user='root', password='Milvus')

    # Check if the connection was successful
    if connections.has_connection('default'):
        print("Connected to Milvus server successfully!")

        # Define fields for the collection
        id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, description="primary id")
        vector_field = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768, description="vector")
        dossier_field = FieldSchema(name="dossier", dtype=DataType.VARCHAR, description="dossier id", max_length=128)
        metadata_field = FieldSchema(name="metadata", dtype=DataType.JSON, description="metadata")

        schema = CollectionSchema(fields=[id_field, dossier_field, vector_field], 
                                  auto_id=True, 
                                  enable_dynamic_field=True, 
                                  description="desc of a collection")

        # Create a collection
        collection = Collection(name="example_collection", schema=schema)
        print("Collection created successfully!")
    else:
        print("Failed to connect to Milvus server.")
except ConnectionNotExistException as e:
    print(f"Connection error: {e}")


# collection_name1 = "tutorial_1"
# collection1 = Collection(name=collection_name1, schema=schema, using='default', shards_num=2)

# # 2. Create a collection
# client.create_collection(
#     collection_name=collection_name1,
#     dimension=768,
#     auto_id=True,
#     schema=schema
# )