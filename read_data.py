
from nomic import embed
import numpy as np
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
import os
load_dotenv()

#read data from data/course_info.json
def read_course_info():
    import json
    with open('data/courses_info.json') as f:
        course_info = json.load(f)
    return course_info

data = read_course_info()
# print(data)

def format_data(data):
    new_data = []
    for d in data:
        new_string = f"{d['name']} taught by {d['faculty']} has class code {d['class_code']} teaches on {d['days']} at {d['start_time']}."
        new_data.append(new_string)
    return new_data

formatted_data = format_data(data)
# print(formatted_data)
print(len(formatted_data))

# {
#     "name": "FINANCIAL INSTITUTIONS AND MARKETS",
#     "faculty": "Tehseen Mazhar Valjee",
#     "start_time": "16:00",
#     "days": "MON - WED",
#     "std_enrolled": 45,
#     "class_limit": "45",
#     "class_code": "97314"
#   },


def embedding_data(data):
    # embeddings = []
    # for d in data:
    # d = data[0]
    # print(d)
    output = embed.text(
        texts=data,
        model='nomic-embed-text-v1.5',
        task_type='search_document',
    )
    # print(output)

    embeddings = [np.array(embedding) for embedding in output['embeddings']]
    # print(embeddings)
    return embeddings
    # print(embeddings[0][0].shape)  # prints: (768,)

embedded_data = embedding_data(formatted_data)
print(len(embedded_data))




pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "nlp"

# if index_name not in pc.list_indexes():
#     pc.create_index(index_name, dimension=768, metric="cosine", spec=ServerlessSpec(
#         cloud="aws",
#         region="us-east-1"
#     ),)

index = pc.Index(index_name)

print("Pinecone Indices: ", pc.list_indexes())

# index = pc.Index(name="test")

for i in range(len(embedded_data)):
    print(i)
    vector = embedded_data[i]
    formatted_sentence = formatted_data[i]
    upsert_response = index.upsert(
    vectors=[
        {
            "id": str(i), # unique string identifier for the vector, must be provided
            "values": vector, # put the embedding vector here
            "metadata": {  # put the actual document's text here
                "text": formatted_sentence,
                # "genre" : "documentary" # other optional metadata
            }
        },
    ],
    namespace="nlp-module-index" # optional, defaults to "default"
)


# Finding similar vectorsq
# q = index.query(
#     namespace="example-namespace",
#     vector=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], # put the query vector here
#     filter={ # optional, to filter the results based on metadata
#         "genre": {"$eq": "documentary"}
#     },
#     top_k=3,
#     include_values=True # optional, to include the vector values in the response
# )
# print(q)
