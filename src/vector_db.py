from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import json

class MilvusDB:
    def __init__(self, uri: str, api_key: str, collection_name: str = "document_embeddings"):
        self.uri = uri
        self.api_key = api_key
        self.collection_name = collection_name
        connections.connect(alias="default", uri=self.uri, token=self.api_key)

    def create_collection(self, dimension: int = 384):
        """
        Create a Milvus collection for storing embeddings.
        """
        fields = [
            FieldSchema(name="chunk_id", dtype=DataType.INT64, is_primary=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
        ]
        schema = CollectionSchema(fields, description="Document Chunk Embeddings")
        collection = Collection(name=self.collection_name, schema=schema)
        print(f"Collection '{self.collection_name}' created.")

    def insert_embeddings(self, embeddings_file: str):
        """
        Insert embeddings into the Milvus collection.
        """
        with open(embeddings_file, "r") as f:
            data = json.load(f)

        collection = Collection(self.collection_name)
        collection.load()
        chunk_ids = [i for i in range(len(data))]
        vectors = [item["embedding"] for item in data]

        collection.insert([chunk_ids, vectors])
        print(f"Inserted {len(vectors)} embeddings.")

    def query(self, query_vector, top_k: int = 5):
        """
        Perform vector similarity search in the collection.
        """
        collection = Collection(self.collection_name)
        collection.load()
        search_params = {"nprobe": 10}
        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
        )
        return results

# Example Usage
if __name__ == "__main__":
    uri = os.getenv("MILVUS_PUBLIC_ENDPOINT")
    api_key = "YOUR_API_KEY"
    milvus = MilvusDB(uri, api_key)

    # Create collection
    milvus.create_collection(dimension=384)

    # Insert embeddings
    embeddings_file = "./embeddings/embeddings.json"
    milvus.insert_embeddings(embeddings_file)

    # Query collection
    dummy_query = [0.1] * 384  # Replace with an actual query embedding
    results = milvus.query(dummy_query, top_k=5)
    for result in results[0]:
        print(f"Chunk ID: {result.id}, Distance: {result.distance}")
