from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
import os
load_dotenv()


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "test"

# if index_name not in pc.list_indexes():
#     pc.create_index(index_name, dimension=8, metric="cosine", spec=ServerlessSpec(
#         cloud="aws",
#         region="us-east-1"
#     ),)

index = pc.Index(index_name)

print("Pinecone Indices: ", pc.list_indexes())

index = pc.Index(name="test")

upsert_response = index.upsert(
    vectors=[
        {
            "id": "vec1", # unique string identifier for the vector, must be provided
            "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], # put the embedding vector here
            "metadata": {  # put the actual document's text here
                "text": "This is a sample document.",
                "genre" : "documentary" # other optional metadata
            }
        },
    ],
    namespace="example-namespace" # optional, defaults to "default"
)

# Finding similar vectorsq
q = index.query(
    namespace="example-namespace",
    vector=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], # put the query vector here
    filter={ # optional, to filter the results based on metadata
        "genre": {"$eq": "documentary"}
    },
    top_k=3,
    include_values=True # optional, to include the vector values in the response
)
print(q)