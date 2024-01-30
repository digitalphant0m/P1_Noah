import app
import sys

array_of_movies = [{"Fellowship of the Ring": "2h 58m", "Two Towers": "2h 59m", "Return of the King": "3h 21m"}]


def main(inputs):
    prompt = f""" Answer the user question on how long a movie runtime is. Filter through Respond with runtime and a quirky response:
    
                User Question: {inputs}
                User Movies: {array_of_movies}
        """
    req_body = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 1000,
            "temperature":0.1
        }
    }

    app.query_llm(req_body)


if __name__ == "__main__":
    main(sys.argv[1])
