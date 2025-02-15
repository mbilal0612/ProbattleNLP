import fitz
import re
from nomic import embed
import numpy as np
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
import os
load_dotenv()

print(fitz.__doc__) 

pdf_path = "data/pa.pdf"
doc = fitz.open(pdf_path)

# Extract text from all pages
text = [page.get_text("text") for page in doc]

# Remove unreadable texts using regular expressions
def remove_unreadable(text):
    return re.sub(r'[^\x20-\x7E]+', '', text)

# Chunk the data so that each embedding has size 768
chunked_text = []
for page_text in text:
    cleaned_text = remove_unreadable(page_text)
    for i in range(0, len(cleaned_text), 768):
        chunked_text.append(cleaned_text[i:i+768])

text = chunked_text

# print(text)

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

embedded_data = embedding_data(text)
print(embedded_data)
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
    formatted_sentence = text[i]
    upsert_response = index.upsert(
    vectors=[
        {
            "id": str(i), # unique string identifier for the vector, must be provided
            "values": vector, # put the embedding vector here
            "metadata": {  # put the actual document's text here
                "text": formatted_sentence,
                "source": "trusted"
                # other optional metadata
            }
        },
    ],
    namespace="nlp-module-index" # optional, defaults to "default"
)
