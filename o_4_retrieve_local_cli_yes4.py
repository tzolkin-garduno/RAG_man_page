import argparse
import pandas as pd
import colorama
from colorama import Fore, Style

from o_3_build_vector_stores import load_model_and_tokenizer_from_pretrained
from langchain_community.vectorstores import FAISS

SEPARATOR=';'
colorama.init()

def load_faiss_store(vector_store_path: str, pretrained_model_path: str):
    embedding_model, _ = load_model_and_tokenizer_from_pretrained(pretrained_model_path)
    faiss_store = FAISS.load_local(
        folder_path=vector_store_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    return faiss_store, embedding_model


def search_query_to_dictlist(faiss_store, embedding_model, query: str, k: int):
    query_vector = embedding_model.embed_query(query)
    docs_and_scores = faiss_store.similarity_search_with_score_by_vector(query_vector, k=k)

    results = []
    for doc, score in docs_and_scores:
        results.append({
            'distance': score,
            'content': doc.page_content,
            'metadata': doc.metadata
        })
    return results


def interactive_query_loop(faiss_store, embedding_model, k: int):
    print("\n🔍 Interactive Query Mode (type 'exit' to quit)\n")
    while True:
        query = input(f"{Fore.YELLOW}Enter your query:{Fore.RESET} ").strip()
        if query.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        results = search_query_to_dictlist(faiss_store, embedding_model, query, k)

        if not results:
            print("No results found.\n")
            continue

        print(f"\nTop {k} results for: '{query}'\n")
        for i, res in enumerate(results, 1):
            print(f"{Fore.BLUE}Result {Style.BRIGHT}{i}:{Style.RESET_ALL}")
            print(f"{Fore.CYAN}  Distance  : {Fore.RESET}{res['distance']:.4f}")
            print(f"{Fore.CYAN}  Metadata  : {Fore.RESET}{res['metadata']}")
            print(f"{Fore.CYAN}  Content   : {Fore.RESET}{res['content'][:300]}{'...' if len(res['content']) > 300 else ''}")
            print(f"{Fore.CYAN}{Style.DIM}-{Fore.RESET}{Style.RESET_ALL}" * 60)


if __name__ == "__main__":
    default_conf_file = "/home/tzolkin/DebConf_2025/FineTune_data/fine_tune_DebConf2025_conf.csv"
    parser = argparse.ArgumentParser(description="Fine-tune a bi-encoder model using contrastive loss.")

    parser.add_argument("--vec_store_conf", type=str, default=default_conf_file,  # required=True,
                        help="File with the configuration specifications of the models to use,"
                             " for the vector store embeddings. The columns are semicolon separated with following headers: "
                             "model_name_or_path;dataset_path;save_path;query_field;doc_field;max_length;batch_size;learning_rate;epochs")

    parser.add_argument("--index", type=int, default=0,
                        help="Index of the vector store to be used from the config file")

    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of hits")
    args = parser.parse_args()

    top_k = args.top_k
    conf_df = pd.read_csv(args.vec_store_conf, sep=SEPARATOR)
    conf_df = conf_df.iloc[args.index, :]
    del args

    print(f"📁 Loading vector store from: {conf_df.vector_store_path}")
    store, embedder = load_faiss_store(vector_store_path=conf_df.vector_store_path,
                                       pretrained_model_path= conf_df.finetuned_model_path)

    interactive_query_loop(store, embedder, top_k)

