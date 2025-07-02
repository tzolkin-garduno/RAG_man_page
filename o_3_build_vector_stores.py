import os
import sys
import argparse
import shutil
import glob
from pathlib import Path

import torch

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from faiss import IndexFlatL2
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig

import pandas as pd


#### EXTRA IMPORTS
"""def check_pythonpath():
    current_dir = os.getcwd()
    utils_dir = os.path.join(current_dir, 'utils')

    if utils_dir not in sys.path:
        sys.path.append(utils_dir)
        print(f"Updated sys.path: {sys.path}")
    else:
        print(f"sys.path already includes {utils_dir}")

check_pythonpath()

from main_process_library  import docs_4_vectorstore_from_chunk_file
"""
###################    END IMPORTS    ####################



################    GLOBALS    #######################
DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
    print("Current CUDA device:", torch.cuda.current_device())
elif torch.backends.mps.is_available():
    x = torch.tensor([1.0, 2.0]).to(torch.device("mps"))
    print("Current device:", x.device)
    DEVICE = 'mps'
else:
    print(f"Current device {DEVICE}")

SEPARATOR=';'

################    END GLOBALS    #######################


################    FUNCTIONS    #######################

def remove_dir(dir_path:str):
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        shutil.rmtree(dir_path)  # Deletes the directory and all its contents


class CustomMeanPoolingEmbedding(Embeddings):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = DEVICE  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def _embed(self, texts:list) -> list[list[float]]:
        # Check if input is list of Documents or strings
        if all(isinstance(t, Document) for t in texts):
            texts = [t.page_content for t in texts]
        elif not all(isinstance(t, str) for t in texts):
            raise ValueError("CustomMeanPoolingEmbedding expects a list of either strings or langchain Documents")

        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        #inputs = {k: v for k, v in inputs.items()}
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = self.model(**inputs)
            embeddings = output.last_hidden_state.mean(dim=1)

        return embeddings.detach().cpu().numpy().tolist()



def load_model_and_tokenizer_from_pretrained(pretrained_model_path:str):
    if "t5" in pretrained_model_path.lower():
        from transformers import T5Tokenizer, T5EncoderModel, T5Config

        tokenizer = T5Tokenizer.from_pretrained(pretrained_model_path)
        model = T5EncoderModel.from_pretrained(pretrained_model_path)
        embedding_dim = T5Config.from_pretrained(pretrained_model_path).d_model
        embedding_model = CustomMeanPoolingEmbedding(model=model, tokenizer=tokenizer)

    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
        model = AutoModel.from_pretrained(pretrained_model_path)
        #embedding_model = HuggingFaceEmbeddings(model_name=pretrained_model_path)
        embedding_dim = AutoConfig.from_pretrained(pretrained_model_path).hidden_size
        embedding_model = CustomMeanPoolingEmbedding(model=model, tokenizer=tokenizer)

    return embedding_model, embedding_dim


def load_vector_store(vector_store_path:str, pretrained_model_path: str):
    def langchain_faiss_exists(vector_store_path: str) -> bool:
        dir_path = Path(vector_store_path)
        return (dir_path / "index.faiss").is_file() and (dir_path / "index.pkl").is_file()

    embedding_model, embedding_dim = load_model_and_tokenizer_from_pretrained(pretrained_model_path=pretrained_model_path)

    if langchain_faiss_exists(vector_store_path=vector_store_path):
        print(f"**** Loading vector store {vector_store_path.split('/')[-1]}")
        vector_store = FAISS.load_local(vector_store_path, embeddings=embedding_model,
                         allow_dangerous_deserialization=True)
    else:
        print(f"**** Creating vector store {vector_store_path.split('/')[-1]}")
        vector_store = FAISS(embedding_function=embedding_model,
                             index=IndexFlatL2(embedding_dim),
                             docstore=InMemoryDocstore(),
                             index_to_docstore_id={})
    vector_store.save_local(vector_store_path)

    return vector_store


def docs_4_vectorstore_from_chunk_file(chunks_csv_file: str,
                                       text_column_name: str = 'chunk') -> list:

    # Load the data from the source CSV file
    source_df = pd.read_csv(chunks_csv_file, sep=SEPARATOR)
    source_df.dropna(inplace=True)

    # Prepare lists for FAISS vectors and metadata
    documents = []

    # Iterate through each row in the source CSV and process the data
    for _, row in source_df.iterrows():

        source_file = os.path.basename(chunks_csv_file)
        chunk_num = row['chunk_num']

        meta = { "chunk_num" : chunk_num,
                 "source_file" : source_file,
               }

        documents += [Document(page_content=row[text_column_name], metadata=meta)]

    return documents


def populate_vector_store_from_finetuned(vector_store_path:str,
                                         finetuned_model_path: str,
                                         files_2_load: list,
                                         text_column: str = 'chunk'):

    vector_store = load_vector_store(vector_store_path=vector_store_path, pretrained_model_path=finetuned_model_path)

    print( f"  **   Populating vector store {vector_store_path}")
    for ind in tqdm(range(len(files_2_load))):
        filename = files_2_load[ind]

        docs_2_add = docs_4_vectorstore_from_chunk_file(chunks_csv_file=filename,
                                                        text_column_name=text_column)

        if docs_2_add:
                vector_store.add_documents(docs_2_add)
    vector_store.save_local(vector_store_path)


def populate_vector_store_from_finetuned_memloss(vector_store_path: str,
                                         finetuned_model_path: str,
                                         files_2_load: list,
                                         text_column: str = 'chunk',
                                         batch_size: int = 32):  # adjust batch size based on your GPU

    vector_store = load_vector_store(vector_store_path=vector_store_path,
                                     pretrained_model_path=finetuned_model_path)

    print(f"  **   Populating vector store {vector_store_path}")

    for filename in tqdm(files_2_load):
        docs_2_add = docs_4_vectorstore_from_chunk_file(chunks_csv_file=filename,
                                                        text_column_name=text_column)

        if not docs_2_add:
            continue

        # Split docs into smaller batches
        for i in range(0, len(docs_2_add), batch_size):
            batch = docs_2_add[i:i + batch_size]
            try:
                vector_store.add_documents(batch)
                torch.cuda.empty_cache()  # clear any residual GPU memory
            except torch.cuda.OutOfMemoryError as e:
                print(f"OOM Error on batch {i // batch_size}: {e}")
                torch.cuda.empty_cache()
                vector_store.save_local(vector_store_path)
                vector_store = load_vector_store(vector_store_path=vector_store_path,
                                                 pretrained_model_path=finetuned_model_path)
                continue  # skip batch or retry logic could be added

    vector_store.save_local(vector_store_path)


def create_vector_stores_from_config(config_csv):
    df = pd.read_csv(config_csv, sep=SEPARATOR)

    for _, row in df.iterrows():
        finetuned_model_path = row['finetuned_model_path']
        vector_store_path = row['vector_store_path']
        chunks_path = row['chunks_path']
        max_length = int(row.get('max_length', 128))
        documents_suffix = row['documents_suffix']
        text_column = row["doc_column"]

        remove_dir(dir_path=vector_store_path)
        os.makedirs(vector_store_path, exist_ok=True)

        print(f"\nProcessing model from: {finetuned_model_path}")

        # Load all chunk data
        chunk_file_list = glob.glob(os.path.join(chunks_path, f'*{documents_suffix}'))
        #print(f"Found {len(chunk_file_list)} chunks to embed.")

        populate_vector_store_from_finetuned_memloss(vector_store_path=vector_store_path,
                                             finetuned_model_path=finetuned_model_path,
                                             files_2_load=chunk_file_list,
                                             text_column=text_column )

        print(f"FAISS index saved to {vector_store_path}")


if __name__=='__main__':

    ##################################################
    #########   Variables
    default_conf_file = "/home/tzolkin/DebConf_2025/FineTune_data/fine_tune_DebConf2025_conf.csv"
    parser = argparse.ArgumentParser(description="Fine-tune a bi-encoder model using contrastive loss.")

    parser.add_argument("--vec_store_conf", type=str, default=default_conf_file,  # required=True,
                        help="File with the configuration specifications of the models to use,"
                             " for the vector store embeddings. The columns are semicolon separated with following headers: "
                             "model_name_or_path;dataset_path;save_path;query_field;doc_field;max_length;batch_size;learning_rate;epochs")

    args = parser.parse_args()

    create_vector_stores_from_config(config_csv=args.vec_store_conf)

