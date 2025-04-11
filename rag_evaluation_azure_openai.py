from dotenv import load_dotenv
import os
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    answer_relevancy,
    faithfulness,
    context_recall,
)
from datasets import Dataset
import pandas as pd
import requests
from openai import AzureOpenAI 
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
import csv

# Load environment variables from .env file
load_dotenv()

# Fetch variables from environment
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_MODEL_DEPLOYMENT = os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT")
AZURE_OPENAI_MODEL_NAME = os.getenv("AZURE_OPENAI_MODEL_NAME")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_EMBEDDING_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_NAME")
AZURE_AI_SEARCH_ENDPOINT = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
AZURE_AI_SEARCH_KEY = os.getenv("AZURE_AI_SEARCH_KEY")
AZURE_AI_SEARCH_INDEX = os.getenv("AZURE_AI_SEARCH_INDEX")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

# Ensure API key is set
if not AZURE_OPENAI_API_KEY:
    raise ValueError("Azure OpenAI API Key is not set in environment variables.")
    
# Set up Azure OpenAI configuration
azure_configs = {
    "base_url": AZURE_OPENAI_ENDPOINT,
    "model_deployment": AZURE_OPENAI_MODEL_DEPLOYMENT,
    "model_name": AZURE_OPENAI_MODEL_NAME,
    "embedding_deployment": AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    "embedding_name": AZURE_OPENAI_EMBEDDING_NAME,
    "api_version" : AZURE_OPENAI_API_VERSION
}    


# Initialize models
def initialize_models():
    """Initialize Azure Chat Model and Embedding Model"""
    azure_model = AzureChatOpenAI(
        openai_api_version=azure_configs["api_version"],
        azure_endpoint=azure_configs["base_url"],
        azure_deployment=azure_configs["model_deployment"],
        model=azure_configs["model_name"],
        validate_base_url=False,
    )

    azure_embeddings = AzureOpenAIEmbeddings(
        openai_api_version=azure_configs["api_version"],
        azure_endpoint=azure_configs["base_url"],
        azure_deployment=azure_configs["embedding_deployment"],
        model=azure_configs["embedding_name"],
    )
    
    return azure_model, azure_embeddings

# Initialize models
azure_model, azure_embeddings = initialize_models()

# Define metrics to use
metrics = [
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
]

# List of test queries
test_queries = [
    "what is business ownership?",
    "what are different types of companies",
    "explain the types of associations"
]

reference_answers = {
    "what is business ownership?": "Business ownership refers to the different legal structures under which a business can be organized and operated. These structures are defined and regulated by law and have various implications for taxation, financing, liability, and management. In Australia, common forms of business ownership are Sole Trader, Partnership, Company, Association, Joint Venture and Trust.",
    "what are different types of companies": "The main types of companies include Public Companies, Proprietary Companies, and limited liability companies.",
    "explain the types of associations": "Associations can be categorized into two main types: incorporated associations and unincorporated associations."
}

# Function to call Azure AI Search
def search_azure_ai(query):
    search_client = SearchClient(
        endpoint=AZURE_AI_SEARCH_ENDPOINT,
        index_name=AZURE_AI_SEARCH_INDEX,
        credential=AzureKeyCredential(AZURE_AI_SEARCH_KEY)
    )
    results = search_client.search(search_text=query, top=5)

    # Print response to debug field names
    #for doc in results:
        #print("Retrieved document:", doc)  # Debugging step

    return [doc.get('Chunk', 'No content found') for doc in results]

# Function to call Azure OpenAI
def call_azure_openai(query, retrieved_docs):
    client = AzureOpenAI(
     api_key=AZURE_OPENAI_API_KEY,  
     api_version=AZURE_OPENAI_API_VERSION,
     azure_endpoint = AZURE_OPENAI_ENDPOINT
    )
    prompt = f"Context:\n{retrieved_docs}\n\nQuestion: {query}\nAnswer:"
    
    response = client.chat.completions.create(
        model=AZURE_OPENAI_MODEL_NAME,
      	messages=[
       		{"role": "system", "content": "You are a helpful assistant."},
        	{"role": "user", "content": prompt}
    	],
        max_tokens=200
    )
    
    return response.choices[0].message.content.strip() 


# Run evaluation
def run_evaluation():
    """Run RAGAS evaluation on test queries."""
    
    results = []
    
    for query in test_queries:
        retrieved_docs = search_azure_ai(query) 
        retrieved_contexts = [doc.strip() for doc in retrieved_docs if isinstance(doc, str)]
        generated_answer = call_azure_openai(query, "\n".join(retrieved_contexts))
        reference_answer = reference_answers.get(query, "")

        dataset = [{
            "question": query,
            "context": " ".join(retrieved_contexts),  # Ensure a single string
            "retrieved_contexts": retrieved_contexts,  # Keep as a list
            "answer": generated_answer,
            "reference": reference_answer
        }]
    
        hf_dataset = Dataset.from_list(dataset)
    
        try:
            print("Running evaluation...")
            evaluation_result = evaluate(hf_dataset, metrics=metrics, llm=azure_model, embeddings=azure_embeddings)
            
            # Print the result to inspect it
            print(f"Evaluation Result for '{query}': {evaluation_result}")
            
            score_dict = evaluation_result.scores[0]
            
            # Store results
            results.append({
                "question": query,
                "ground_truth": reference_answer,
                "answer": generated_answer,
                "contexts": "\n".join(retrieved_contexts),
                "faithfulness": float(score_dict.get("faithfulness", 0)),
                "answer_relevancy": float(score_dict.get("answer_relevancy", 0)),
                "context_recall": float(score_dict.get("context_recall", 0)),
                "context_precision": float(score_dict.get("context_precision", 0))
            })
        
        except Exception as e:
            print(f"Error during evaluation for query '{query}': {e}")
   
   # Convert to DataFrame
    result_df = pd.DataFrame(results)
    print("Final Evaluation Results:")
    print(result_df)

    # Save results to CSV
    result_df.to_csv("evaluation_results.csv", index=False)

# Run evaluation
evaluation_results = run_evaluation()