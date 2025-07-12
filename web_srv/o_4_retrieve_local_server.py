import argparse
import pandas as pd
import json
import readline
import socket

import sys
sys.path.append('/home/tzolkin/Documents/UTT/Ecosystems_Code/DebConf_2025_Code')

from o_3_build_vector_stores import load_model_and_tokenizer_from_pretrained
from langchain_community.vectorstores import FAISS

SEPARATOR=';'

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

def get_pkgs_idx():
    fh = open('/home/tzolkin/man-mirror/pkg_by_man.json', 'r')
    return json.loads(fh.read())

def handle_conn(connection, client, faiss_store, embedding_model, k: int):
    pkg_idx = get_pkgs_idx()
    qry = connection.recv(256).decode("utf-8")
    print(f'Client query: {qry}')
    results = search_query_to_dictlist(faiss_store, embedding_model, qry, k)

    if not results:
        connection.send(b"No results found\n")
        return nil

    connection.send(b"<div class=\"resultset\">\n")
    for i, res in enumerate(results, 1):
        connection.send(b"\n<table class=\"result\">\n")
 
        # Which manpage are we getting this result chunk from?
        manpage = res['metadata']['source_file']

        # Which package can we find the manpage in?
        # We are dealing only with "section 1" manuals, we can temporarily hardcode
        # it into our filter.
        pkgs = None
        try:
            man_idx = manpage.index('.1_chunks.csv')
            manpage = manpage[0:man_idx]
            abs_manpage = manpage + '.1'

            if abs_manpage in pkg_idx.keys():
                pkgs = ', '.join(pkg_idx[abs_manpage])
            else:
                print(f'-!- {manpage} → {abs_manpage}')

        except ValueError:
            manpage = '* cannot determine! *'

        connection.send(b'<tr class="package"><th>Package:</th><td>' )
        if pkgs:
            connection.send(pkgs.encode('utf-8'))
        else:
            connection.send(b'* Unknown! (bug?) *')
        connection.send(b"</td></tr>\n")

        man_url = f'https://manpages.debian.org/jump?q={manpage}'

        connection.send(b'<tr class="manpage"><th>Manpage</th><td>' +
                        f'<a href="{man_url}">'.encode('utf-8') +
                        manpage.encode('utf-8') +
                        b"(1)</a></td></tr>\n")

        # What is the vectorial distance?
        distance = res['distance']
        if distance <= 2:
            dist = 'dist-short'
        elif distance <= 4:
            dist = 'dist-medium'
        else:
            dist = 'dist-long'
        connection.send(f'<tr class="distance {dist}"><th>Distance</th><td>'.encode('utf-8') +
                        f"{distance:.4f}</td></tr>\n".encode('utf-8'))
        
        # Do we have any metadata to support this?
        connection.send(b'<tr class="metadata"><th>Metadata</th><td>')
        connection.send(str(res['metadata']).encode('utf-8'))
        connection.send(b"</td></tr>\n")

        # Chunk content
        connection.send(b'<tr class="chunk"><th>Content chunk</th><td>')
        if len(res['content']) > 300:
            content = res['content'][:300] + ' (...)'
        else:
            content = res['content']
        connection.send(content.encode('utf-8'))
        connection.send(b"</td><tr>\n")
        connection.send(b"</table>\n")
    connection.send(b"</div>\n")
    connection.close()

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
    parser.add_argument("--tcp_port", type=int, default=31337,
                        help="TCP port to listen on")
    args = parser.parse_args()

    # Set up TCP socket
    tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_port = args.tcp_port
    server_address = ('localhost', tcp_port)
    tcp_socket.bind(server_address)
    tcp_socket.listen(1)
    print(f'Waiting for connections on TCP port {tcp_port}')

    top_k = args.top_k
    conf_df = pd.read_csv(args.vec_store_conf, sep=SEPARATOR)
    conf_df = conf_df.iloc[args.index, :]
    del args

    print(f"📁 Loading vector store from: {conf_df.vector_store_path}")
    store, embedder = load_faiss_store(vector_store_path=conf_df.vector_store_path,
                                       pretrained_model_path= conf_df.finetuned_model_path)

    while True:
        connection, client = tcp_socket.accept()
        handle_conn(connection, client, store, embedder, top_k)
