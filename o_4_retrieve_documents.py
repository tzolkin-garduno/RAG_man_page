import os
import argparse
import pandas as pd
import shutil
import numpy as np
from tqdm import tqdm
from o_3_build_vector_stores import load_model_and_tokenizer_from_pretrained
from langchain_community.vectorstores import FAISS

SEPARATOR = ';'


def load_faiss_store(vector_store_path: str, pretrained_model_path: str):
    embedding_model, _ = load_model_and_tokenizer_from_pretrained(pretrained_model_path)
    faiss_store = FAISS.load_local(
        folder_path=vector_store_path,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
    return faiss_store, embedding_model


def search_query_to_df(faiss_store, embedding_model, query: str, k: int):
    query_vector = embedding_model.embed_query(query)
    docs_and_scores = faiss_store.similarity_search_with_score_by_vector(query_vector, k=k)

    results = []
    for doc, score in docs_and_scores:
        results_dict = {k: v for k, v in doc.metadata.items()}
        results_dict.update({'distance': score, 'chunk': doc.page_content, 'query': query})
        results.append(results_dict)

    results = pd.DataFrame(results)
    return results


def remove_dir(dir_path:str):
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        shutil.rmtree(dir_path)  # Deletes the directory and all its contents


def retrieve_multiple_queries(query_list: list[str],
                              faiss_store, embedding_model, k: int, save_to_path: str = None):
    remove_dir(dir_path=os.path.dirname(save_to_path))
    os.makedirs(os.path.dirname(save_to_path), exist_ok=True)
    output_df = pd.DataFrame()
    for query in tqdm(query_list, total=len(query_list)):
        temp_df = search_query_to_df(query=query, faiss_store=faiss_store, embedding_model=embedding_model, k=k)
        output_df = pd.concat([output_df, temp_df])

    output_df.to_csv(save_to_path, sep=SEPARATOR, index=False)
    return output_df



def generating_all_retrieval_docs(conf_df: pd.DataFrame):

    for i, row in conf_df.iterrows():
        print(f"📁 Loading vector store from: {row.vector_store_path}")
        store, embedder = load_faiss_store(vector_store_path=row.vector_store_path,
                                           pretrained_model_path=row.finetuned_model_path)
        query_list = pd.read_csv(row.ground_truth_data_file, sep=SEPARATOR)
        query_list = query_list.loc[:, row.query_field]
        retrieve_multiple_queries(query_list=query_list, faiss_store=store, embedding_model=embedder,
                                  k=row.top_k,
                                  save_to_path=row.retrieved_docs_file)


if __name__ == "__main__":
    default_conf_file = "/home/tzolkin/DebConf_2025/FineTune_data/fine_tune_DebConf2025_conf.csv"
    parser = argparse.ArgumentParser(description="Fine-tune a bi-encoder model using contrastive loss.")

    parser.add_argument("--vec_store_conf", type=str, default=default_conf_file,  # required=True,
                        help="File with the configuration specifications of the models to use,"
                             " for the vector store embeddings. The columns are semicolon separated with following headers: "
                             "model_name_or_path;dataset_path;save_path;query_field;doc_field;max_length;batch_size;learning_rate;epochs")

    parser.add_argument("--index", type=int, default=3,
                        help="Index of the vector store to be used from the config file")

    args = parser.parse_args()

    conf_df = pd.read_csv(args.vec_store_conf, sep=SEPARATOR)
    generating_all_retrieval_docs(conf_df=conf_df)
