import util


def create_chunks():
    text = util.extract_text_from_pdf('./documents/LordoftheRings-FOTR.pdf')

    chunks = util.split_text_into_sentences(text)

    embeddings = util.query_hf_embedded(chunks)

    util.store_embeddings(embeddings, chunks)
