from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import json


class MilvusDB:
    def __init__(self, uri: str, api_key: str, cluster_name: str = "default"):
        """
        Initialize connection to Milvus Cloud.
        :param uri: Milvus public endpoint
        :param api_key: Milvus Cloud API key
        :param cluster_name: Collection alias (default: "default")
        """
        connections.connect(
            alias=cluster_name,
            uri=uri,
            token=api_key,  # Use your Milvus Cloud API Key
        )
        self.collection_name = "document_embeddings"

    def create_collection(self):
        """
        Create Milvus collection with schema for storing embeddings.
        """
        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),  # Use 384 dimensions for MiniLM
        ]
        schema = CollectionSchema(fields, "Document Embeddings")
        Collection(self.collection_name, schema)

    def insert_embeddings(self, embeddings_file: str):
        """
        Insert precomputed embeddings into the Milvus collection.
        """
        with open(embeddings_file, "r") as f:
            data = json.load(f)

        collection = Collection(self.collection_name)
        collection.load()  # Load collection into memory

        vectors = [item["embedding"] for item in data]
        chunk_ids = [i for i in range(len(vectors))]

        # Insert into Milvus
        collection.insert([chunk_ids, vectors])
        print(f"Inserted {len(vectors)} vectors into Milvus.")

    def query_vectors(self, query_embedding, top_k: int = 5):
        """
        Perform vector search on the Milvus collection.
        """
        collection = Collection(self.collection_name)
        collection.load()
        search_params = {"nprobe": 10}  # Milvus search parameter
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
        )
        return results


# Example: Create collection, insert embeddings, and query
if __name__ == "__main__":
    uri = "https://in03-2c3b8bd41a90c08.serverless.gcp-us-west1.cloud.zilliz.com"
    api_key = "f65478fc22acefefdb96955662fcbe5aa470a9d64e847d59b86a296f8fec1e54da5be7704857c10afcafe7a5eb9d106d947e6857"  # Replace with your actual API key
    db = MilvusDB(uri, api_key)

    # Step 1: Create a collection in Milvus Cloud
    db.create_collection()

    # Step 2: Insert embeddings into Milvus
    embeddings_file = "./embeddings/embeddings.json"  # Path to your generated embeddings file
    db.insert_embeddings(embeddings_file)

    # Step 3: Query embeddings
    dummy_query_embedding = [0.1] * 384  # Replace with a real query embedding
    results = db.query_vectors(dummy_query_embedding)
    for result in results[0]:
        print(f"Found: {result.id}, Distance: {result.distance}")
