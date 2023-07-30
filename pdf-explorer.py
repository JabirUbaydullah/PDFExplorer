import streamlit as st
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from llama_cpp import Llama
import torch

MAX_CTX = 2048
WINDOW_SIZE = 128
STEP_SIZE = 100
TOP_K = 32
N_THREADS = 10
N_GPU_LAYERS = 20000

PROMPT_BEGIN = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n"

PROMPT = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to
make up an answer. Don't make up new terms which are not available in the context.

Context:
{context}

"""

PROMPT_END = """
### Instruction: {prompt}

### Response:
"""

mps_device = torch.device("mps")


def process_pdf(file_name):
    print("Analyzing", file_name)

    text = extract_text(file_name)
    text = " ".join(text.split())
    text_tokens = text.split()

    sentences = []
    for i in range(0, len(text_tokens), STEP_SIZE):
        window = text_tokens[i : i + WINDOW_SIZE]
        if len(window) < WINDOW_SIZE:
            break
        sentences.append(window)

    paragraphs = [" ".join(s) for s in sentences]
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    model.max_seq_length = 384
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    embeddings = model.encode(
        paragraphs,
        show_progress_bar=True,
        convert_to_tensor=True,
    )
    embeddings.to(mps_device)

    llm = Llama(
        model_path="./models/nous-hermes-llama2-13b.ggmlv3.q4_K_M.bin",
        n_ctx=MAX_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=False,
    )

    return model, cross_encoder, embeddings, paragraphs, llm


def user_query(user_query, embeddings, text, model, cross_encoder):
    query_embeddings = model.encode(user_query, convert_to_tensor=True)
    query_embeddings = query_embeddings.to(mps_device)
    search_results = util.semantic_search(
        query_embeddings,
        embeddings,
        top_k=TOP_K,
    )[0]

    cross_input = [
        [user_query, text[search_result["corpus_id"]]]
        for search_result in search_results
    ]
    cross_scores = cross_encoder.predict(cross_input)

    for idx in range(len(cross_scores)):
        search_results[idx]["cross_score"] = cross_scores[idx]

    results = []
    search_results = sorted(
        search_results, key=lambda x: x["cross_score"], reverse=True
    )
    for search_result in search_results[:5]:
        results.append(text[search_result["corpus_id"]].replace("\n", " "))
    return results


def run_query(query):
    print("Running query:", query)

    results = user_query(
        query,
        st.session_state["embeddings"],
        st.session_state["paragraphs"],
        st.session_state["model"],
        st.session_state["cross_encoder"],
    )
    context = "\n".join(results)

    post_prompt = PROMPT_END.format(prompt=query)
    prompt = PROMPT.format(
        context=context[: (MAX_CTX * 4 - len(post_prompt) - len(PROMPT_BEGIN))]
    )

    query = PROMPT_BEGIN + prompt + post_prompt

    output = st.session_state["llm"](query, max_tokens=512, stop=["### Response:"])
    st.write(output["choices"][0]["text"].strip())


def main():
    st.title("PDFExplorer")
    file_name = st.file_uploader("Upload a PDF file", type=["pdf"])

    if file_name is not None:
        if (
            "file_name" not in st.session_state
            or st.session_state["file_name"] != file_name.name
        ):
            st.session_state["file_name"] = file_name.name
            (
                st.session_state["model"],
                st.session_state["cross_encoder"],
                st.session_state["embeddings"],
                st.session_state["paragraphs"],
                st.session_state["llm"],
            ) = process_pdf(file_name)

        form = st.form(key="my_form")
        query = form.text_input(
            "Ask a question", placeholder="What is the meaning of life?"
        )
        submit_button = form.form_submit_button(label="Submit")
        if submit_button:
            run_query(query)


if __name__ == "__main__":
    main()
