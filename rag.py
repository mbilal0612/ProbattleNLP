import re
from dotenv import load_dotenv
import os
from langchain_nomic import NomicEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv(override=True) # Load environment variables from .env file, override any existing variables

# Making a Langchain Embeddings Object using Nomic

embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")

# Making a Pinecone Vector Store Object

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "nlp"  # change if desired
index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings, namespace="nlp-module-index")

# Making a Retriever Object (Allows you to find similar documents in your Pinecone index, given a query)

retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 15, "score_threshold":0.80 ,"filter": {"source": "trusted"}},
)

# Making a ChatGroq Object (This is the LLM model that will generate responses)

llm = ChatGroq(model="llama3-8b-8192", stop_sequences= None, temperature=0)

# Function to format the retrieved documents, gotten from the retriever

def format_docs(docs):
    print("docs:", docs)
    print()
    return "\n\n".join(f"{i+1}. {doc.page_content}" for i, doc in enumerate(docs))

# Function to validate the query

def validate_query(query):
    banned_patterns = [
        r"ignore previous instructions",
        r"repeat after me",
        r"act as an AI developer",
        r"write an unrestricted version",
        r"tell me how to bypass",
    ]

    if len(query) < 5 or len(query) > 200:
        raise ValueError("Query length out of bounds.")
    if any(word in query.lower() for word in ["ignore", "system", "prompt", "admin"]):
        raise ValueError("Potential prompt injection detected.")
    for pattern in banned_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            raise ValueError("Banned pattern detected in query.")
    return query


# Making a custom prompt which had two variables, "context" and question

# Note: This prompt_template expects a dictionary/JSON with the keys "context" and "question" as input

prompt_template = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["context", "question"],
            template=( # The constructed prompt from the variables
                "You are a strict assistant only answer questions based on the data provided."
                "You are an assistant for question-answering tasks. Use the following "
                "pieces of retrieved context to answer the question. If you don't know "
                "the answer, just say that you don't know."
                "and keep the answer concise.\n\n"
               
                
                "Question: {question}\n"
                "Context: {context}\n"
                "Answer:"
            )
            
        )
    )
])

# A simple function that logs the input and returns it

def logger(input):
    print(input)
    return input

# A chain with the modified prompt

rag_chain = (
    # The starting input to all these is passed in the invoke function
    # e.g rag_chain.invoke("Tell me about the paper: Attention is all you Need")
    
    # The first runnable in the chain:
    {
        "context": retriever | format_docs | logger,
        "question": validate_query |RunnablePassthrough()
    }
    # It makes a dictionary using the input
    # the input is passed through the validate_query function, then the retriever, then the format_docs function, then the logger function
    # the validate_query function validates the query, the retriever finds similar documents in the Pinecone index, and the format_docs function formats them
    # the logger function logs the input and returns it, 
    # RunnablePassthrough is a simple function that returns the input,
    # which means the 
    
    # The second runnable constructs the prompt using the previous runnables' output
    # The previous runnables' output is is a dictionary: {"context": ..., "question": ...}
    | prompt_template
    # This makes a prompt using the context and question from the previous runnables' output
    # The prompt is just a large string that is passed to the next runnable
    
    # The third runnable is the LLM model that generates the response
    | llm
    # The LLM model generates a response using the prompt
    # It's like passing something to the ChatGPT model and getting a response
    
    # The fourth runnable is the output parser that converts the output to a string
    | StrOutputParser() 
    # The llm's output is a dictionary with several fields
    # The output parser takes the response.content field and returns it as a string
)

# The chain simply looks like this:
rag_chain = (
    {
        "context": retriever | format_docs | logger,
        "question": validate_query | RunnablePassthrough()
    }
    | prompt_template
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("Give a list of teachers teaching DATA STRUCTURES")