import os
import pandas as pd
import openai
import nltk
import re

from pandas.core.methods.describe import describe_timestamp_1d
from transformers import AutoTokenizer
from nltk.tokenize import sent_tokenize
import random
from math import ceil
from tqdm import tqdm
import argparse
import time
import backoff

nltk.download('punkt') # Download punkt tokenizer for sentence splitting


with open(os.path.join(os.environ.get("HOME"), ".openai", "Maps-Firms-key"), 'r') as f:
    OPENAI_KEY = f.readline().strip()

openai.api_key = OPENAI_KEY
SEPARATOR = ';'
TOKENIZER_CHECKPOINT = "sentence-transformers/all-MiniLM-L6-v2"


def semantic_chunking_csv(source_file_path: str,
                          target_file_path: str,
                          max_token_length:int = None) ->  pd.DataFrame:
    with open(source_file_path, 'r') as f:
        paragraphs = f.readlines()

    buffer_tokens = 10
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_CHECKPOINT)
    if max_token_length is None:
        max_token_length = tokenizer.model_max_length - buffer_tokens
    else:
        max_token_length = max_token_length - buffer_tokens

    chunks = []
    current_chunk = ""

    def add_chunk(text):
        chunks.append({
            "chunk": text.strip()
        })


    for para in paragraphs:
        if not para.strip():
            continue

        block = " ".join(sent_tokenize(para.strip()))
        tentative = current_chunk + "\n\n" + block if current_chunk else block
        tokenized_len = len(tokenizer.encode(tentative, truncation=False))

        if tokenized_len <= max_token_length:
            current_chunk = tentative
        else:
            if current_chunk:
                add_chunk(current_chunk)
            if len(tokenizer.encode(block)) > max_token_length:
                sentences = sent_tokenize(block)
                sentence_block = ""

                for sentence in sentences:
                    tentative_sentence = sentence_block + " " + sentence if sentence_block else sentence
                    if len(tokenizer.encode(tentative_sentence)) <= max_token_length:
                        sentence_block = tentative_sentence
                    else:
                        add_chunk(sentence_block)
                        sentence_block = sentence

                if sentence_block:
                    add_chunk(sentence_block)
                current_chunk = ""
            else:
                current_chunk = block

    # Final chunk
    if current_chunk:
        add_chunk(current_chunk)

    chunks_df = pd.DataFrame(chunks)
    chunks_df['chunk_num'] = [i for i in range(1,len(chunks_df)+1)]
    chunks_df.to_csv(target_file_path, header=True, index=False, sep=SEPARATOR)
    return chunks_df


def generate_query(chunk: str, prompt:str) -> str:
    """Generate a question based on the given chunk of text."""
    extended_prompt=f"{prompt} {chunk}"
    # Call OpenAI to generate a question
    response = openai.chat.completions.create(
        model="gpt-4o-mini",  #"text-davinci-003",  # You can choose another model if you prefer
        messages=[{"role": "user", "content": extended_prompt}],
        max_tokens=100,
        temperature=0.7
    )

    response_pre = response.choices[0]
    question = response_pre.message.content
    return question


@backoff.on_exception(backoff.expo,
                      Exception,
                      max_time=10)
def generate_query_with_retry(chunk, prompt):
    return generate_query(chunk, prompt)



def create_chunks(dir_path: str, chunk_file_suffix: str):
    ls_dir = os.listdir(dir_path)
    print(f" ****   Creating chunk files in {dir_path}")
    all_files = [[f, f.split('.txt')[0]+chunk_file_suffix ] for f in ls_dir if f.endswith('.txt')]
    all_files = pd.DataFrame(all_files, columns=['original_file', 'chunk_file'])

    for _, row in tqdm(all_files.iterrows(), total=len(all_files)):
            if row.chunk_file not in ls_dir:
                semantic_chunking_csv(source_file_path=os.path.join(dir_path, row.original_file),
                                      target_file_path=os.path.join(dir_path, row.chunk_file),
                                      max_token_length=150)


def extract_between_name_and_synopsis(man_file: str =None, man_text:str = None) -> str:
    if man_file:
        with open(man_file, 'r') as f:
            man_text = ''
            for _ in range(30):
                man_text += f.readline()

    match = re.search(
        r"\s*NAME\s*(.*?)\s*SYNOPSIS",
        man_text,
        re.DOTALL | re.IGNORECASE
    )
    if match:
        content = match.group(1).strip()
        return content
    else:
        return ""


def create_preamble(dir_path: str, chunk_file_suffix: str):

    # This function adds the description of the man page right to the
    # beginning of each chunk, this is what is called "preamble" in this function
    # Then it makes the
    ls_dir = os.listdir(dir_path)
    print(f" ****   Adding Descriptions to Chunks in {dir_path}")
    names = []
    for filename in tqdm(ls_dir, total=len(ls_dir)):
        if filename.endswith(".txt"):
            names += [[filename, extract_between_name_and_synopsis(man_file=os.path.join(dir_path,filename))]]

    names_df = pd.DataFrame(names, columns=['filename', 'description'])
    del names

    for filename in tqdm(ls_dir, total=len(ls_dir)):
        if filename.endswith(chunk_file_suffix):
            chunks_df = pd.read_csv(os.path.join(dir_path,filename), sep=SEPARATOR)
            no_chunks_file = filename.split(chunk_file_suffix)[0] + '.txt'
            filtered = names_df['description'].loc[names_df['filename'] == no_chunks_file]
            if not filtered.empty:
                description = filtered.iloc[0]
            else:
                description = ''  # or handle the missing case appropriately
            chunks_df['simple_chunk'] = chunks_df['chunk']
            chunks_df['preamble'] = [description for _ in range(len(chunks_df))]
            chunks_df['chunk'] = 'COMMAND: ' + chunks_df['preamble']+ ' PARTIAL TEXT: ' + chunks_df['simple_chunk']

            chunks_df.to_csv(os.path.join(dir_path, filename), sep=SEPARATOR, index=False)


def create_preamble_training(training_data_file:str, dir_path: str):
    training_data = pd.read_csv(training_data_file, sep=SEPARATOR)
    preamble_list =[]
    for _, row in tqdm(training_data.iterrows(), total=len(training_data)):
        description = pd.read_csv(os.path.join(dir_path, row.source_file), sep=SEPARATOR)
        description = description.loc[0,'simple_chunk']
        preamble = extract_between_name_and_synopsis(man_text=description)
        preamble_list.append(preamble)

    training_data['preamble'] = preamble_list

    training_data['simple_chunk'] = training_data['chunk']
    training_data['chunk'] = 'COMMAND: ' + training_data['preamble'] + ' PARTIAL TEXT: ' + training_data['simple_chunk']

    training_data.to_csv(training_data_file, sep=SEPARATOR, index=False)


def create_queries(data_source_path: str,
                   save_to_file:str,
                   data_distribution_conf: str,
                   sample_frac: float,
                   chunk_file_suffix: str,
                   start_from: int = 0):

    training_conf_df = pd.read_csv(data_distribution_conf, sep=SEPARATOR, index_col='label')
    qa_options = list(training_conf_df.index)
    weights = list(training_conf_df.loc[:,'weight'])

    chunk_files = [f for f in os.listdir(data_source_path) if f.endswith(chunk_file_suffix)]
    sample_size = max(1, int(len(chunk_files) * sample_frac))  # Use max to ensure at least 1 item if the list is small
    chunk_files = random.sample(chunk_files, sample_size)

    columns = ["chunk", "query", "qa_type", "chunk_num", "source_file"]

    for ind, filename in tqdm(enumerate(chunk_files, start=start_from), total=len(chunk_files)):
        try:
            # print(f"{ind}.   Processing command:    {filename.split('.')[0]}")
            file_path = os.path.join(data_source_path, filename)
            chunks_df = pd.read_csv(file_path, sep=SEPARATOR)
            chunks_df.index = [i for i in range(2, len(chunks_df) + 2)]
            output_data = []

            if len(chunks_df) >= ceil(1 / sample_frac):
                random_rows = chunks_df.sample(frac=sample_frac)
            elif len(chunks_df) > 1:
                random_rows = chunks_df.sample(n=1)
            else:
                continue

            for chunk_num, row in random_rows.iterrows():
                # print(f"Chunk {chunk_num}")
                chunk = row['chunk']
                qa_type = random.choices(qa_options, weights=weights, k=1)[0]
                prompt = training_conf_df.loc[qa_type, 'prompt']

                #question = generate_query_with_retry(chunk=chunk, prompt=prompt)  # Generate the question for the chunk
                question = generate_query(chunk=chunk, prompt=prompt)  # Generate the question for the chunk
                output_data.append([chunk, question, qa_type, chunk_num, filename])  # Add to the output list

            df = pd.DataFrame(output_data, columns=columns)
            df.to_csv(save_to_file, mode='a', sep=SEPARATOR, index=False)

        except Exception as e:
            print(f"Skipping index {ind} due to error: {e}")
            time.sleep(10)
            continue

    print(f"Chunks and queries saved to {save_to_file}.")


if __name__=='__main__':

    default_conf_file = "/home/tzolkin/DebConf_2025/FineTune_data/fine_tune_DebConf2025_conf.csv"

    parser = argparse.ArgumentParser(description="Generating data for fine-tunin a bi-encoder model using contrastive loss."
                                                 "Also generating data for evaluation of the results of the RAG process")

    parser.add_argument("--conf_file", type=str, default=default_conf_file,  # required=True,
                        help="File with the configuration specifications of the paths to where the data is to be saved to,"
                             " the columns are semicolon separated with following headers: "
                             "model_name_or_path;finetuned_model_path;chunks_path;dataset_path;vector_store_path;retrieved_docs_path;generation_path;query_field;doc_column;documents_suffix;max_length;batch_size;learning_rate;epochs;top_k")

    parser.add_argument("--start_from", type=int, default=-1)

    args = parser.parse_args()

    conf_df = pd.read_csv(args.conf_file, sep=SEPARATOR)
    conf_df = conf_df.iloc[0,:]
    start_from = args.start_from if args.start_from>=0 else 0

    data_source_file = conf_df['chunks_path']
    chunk_file_suffix = conf_df["documents_suffix"]
    training_data_file = conf_df["training_data_file"]
    chunks_path = conf_df['chunks_path']


    #create_chunks(dir_path=data_source_file, chunk_file_suffix=chunk_file_suffix)
    #create_preamble(dir_path=data_source_file, chunk_file_suffix=chunk_file_suffix)

    """
    # Generating data for fine tuning
    
    training_data_file = conf_df['training_data_file']
    training_conf_file = "/home/tzolkin/DebConf_2025/FineTune_data/training_conf.csv"
    print(f"*** Creating training data. Saving to {training_data_file}")
    create_queries(data_source_file,
                   save_to_file=training_data_file,
                   data_distribution_conf=training_conf_file,
                   sample_frac=0.01,
                   start_from=start_from,
                   chunk_file_suffix=chunk_file_suffix)
    """

    #create_preamble_training(dir_path=chunks_path, training_data_file=training_data_file)


    # Creating data for evaluation, that is, the ground truth
    ground_truth_file = conf_df['ground_truth_data_file']
    ground_truth_conf_file = "/home/tzolkin/DebConf_2025/FineTune_data/ground_truth_positive_negative_conf.csv"
    print(f"*** Creating ground truth data. Saving to {ground_truth_file}")
    create_queries(data_source_file,
                   save_to_file=ground_truth_file,
                   data_distribution_conf=ground_truth_conf_file,
                   sample_frac=0.02,
                   chunk_file_suffix=chunk_file_suffix)

