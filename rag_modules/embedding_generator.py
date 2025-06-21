from langchain_community.embeddings import HuggingFaceBgeEmbeddings, OpenAIEmbeddings


def generate_embeddings(chunks, embedding_type="huggingface"):
    if embedding_type == "openai":
        embeddings = OpenAIEmbeddings()
    elif embedding_type == "huggingface":
        embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-m3")
    else:
        raise ValueError("Invalid embedding type specified.")
    return embeddings.embed_documents([chunk.page_content for chunk in chunks])
