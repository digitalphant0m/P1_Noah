import requests
import os
from dotenv import load_dotenv

load_dotenv()


def query_llm(payload):
    api_url = os.getenv('API_URL')
    api_key = os.getenv('API_KEY')

    req_header = {"Authorization": f"Bearer {api_key}"}

    response = requests.post(api_url, headers=req_header, json=payload)

    llm_data = response.json()

    print(llm_data[0]['generated_text'] + '\n\n')


def query_custom(prompt):
    print('hello custom')
