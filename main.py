import sys
import chromadb
import util

client = chromadb.Client()

collection = client.get_or_create_collection(name="library")


def main(user_input):
    question = " ".join(user_input)
    embedded_question = util.query_hf_embedded([question])

    result = collection.query(
        query_embeddings=embedded_question,
        n_results=2,
    )

    prompt = f""" You are a helpful assistant who remembers lines from a movie transcript, based on the provided context:

                Context: {result["documents"]},
                User Question: {question}
        """
    req_body = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 1000,
            "temperature": 0.1
        }
    }

    util.query_llm(req_body)


if __name__ == "__main__":
    main(sys.argv[1:])
