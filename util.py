import requests
import os
from dotenv import load_dotenv
import nltk
from PyPDF2 import PdfReader
import chromadb

load_dotenv()
nltk.download('punkt')

client = chromadb.Client()
collection = client.get_or_create_collection(name="library")


def query_hf_embedded(texts):
    print("\n", texts, "\n")
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{os.getenv('MODEL_ID')}"

    req_header = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}

    response = requests.post(api_url, headers=req_header, json={"inputs": texts, "options": {"wait_for_model": True}})

    return response.json()


def query_llm(payload):
    api_url = os.getenv('API_URL')
    api_key = os.getenv('API_KEY')

    req_header = {"Authorization": f"Bearer {api_key}"}

    response = requests.post(api_url, headers=req_header, json=payload)

    llm_data = response.json()

    print(llm_data[0]['generated_text'] + '\n\n')


def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf = PdfReader(file)
        text = " ".join(page.extract_text() for page in pdf.pages)
    return text


def split_text_into_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return sentences


def store_embeddings(embeddings, chunks):
    embed_ids = []
    for count, ele in enumerate(embeddings):
        embed_id = "doc" + str(count)
        embed_ids.append(embed_id)

    collection.add(
        embeddings=embeddings,
        documents=chunks,
        ids=embed_ids
    )
