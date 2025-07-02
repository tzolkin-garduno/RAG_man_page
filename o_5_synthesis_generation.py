import os
import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm
import argparse

from o_4_retrieve_documents import remove_dir


SEPARATOR = ';'
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


def synthesize_responses(conf_values,
                         classifier_model: str,
                         summarizer_model: str ):
    """
    Reads a semicolon-separated file with columns [chunk_num, source_file, distance, chunk, query],
    filters for relevant chunk per query using zero-shot classification,
    and generates a synthesis (summary) of the filtered chunk for each query.

    Args:
        input_file (str): Path to the input .csv file (semicolon-separated).
        output_file (str): Path to write the output CSV with columns [query, summary].
        classifier_model (str): Model name for zero-shot classification.
        summarizer_model (str): Model name for summarization.
    """
    # Load data
    df = pd.read_csv(conf_values.retrieved_docs_file, sep=SEPARATOR)

    # Initialize pipelines
    classifier = pipeline("zero-shot-classification", model=classifier_model, device=DEVICE)
    summarizer = pipeline("summarization", model=summarizer_model, device=DEVICE)

    candidate_labels = ["yes", "no"]
    relevant_class = []
    summaries = []
    score_yes = []
    score_no = []

    for i in tqdm(range(len(df))):
        row = df.loc[i]
        text = row[conf_values.doc_column]
        query = row[conf_values.query_field]

        #hypothesis = f"This text is {{}} to the query: '{query}'"
        hypothesis = f"Determine if the text answers the question: '{query}' – {{}}"
        cls = classifier(text, candidate_labels, hypothesis_template=hypothesis, multi_label=True)
        scores = dict(zip(cls["labels"], cls["scores"]))
        score_yes.append(scores["yes"])
        score_no.append(scores["no"])
        if scores["yes"] > scores["no"] and scores['yes']>0.6:
            relevant_class.append('yes')
            summary_output = summarizer(text, max_length=150, min_length=40, do_sample=False)
            summaries.append(summary_output[0]['summary_text'])

        else:
            relevant_class.append('no')
            summaries.append('')

    df['qa_type'] = relevant_class
    df['qa_type'] = df['qa_type'].replace('yes', 'positive')
    df['qa_type'] = df['qa_type'].replace('no', 'negative')
    df['positive_score'] = score_yes
    df['negative_score'] = score_no
    df['synthesis'] = summaries

    df.to_csv(conf_values.generated_responses_file, index=False, sep=SEPARATOR, header=True)




def generating_all_synthesis(conf_df: pd.DataFrame, classifier_model:str, summarizer_model:str):

    for i, row in conf_df.iterrows():
        print(f"📁 Loading retrieved docs from: {row.retrieved_docs_file}")
        remove_dir(dir_path=os.path.dirname(row.generated_responses_file))
        os.makedirs(os.path.dirname(row.generated_responses_file), exist_ok=True)

        synthesize_responses(conf_values=row,
                             classifier_model=classifier_model, summarizer_model=summarizer_model)

if __name__ == '__main__':


    default_conf_file = "/home/tzolkin/DebConf_2025/FineTune_data/fine_tune_DebConf2025_conf.csv"
    parser = argparse.ArgumentParser(description="Fine-tune a bi-encoder model using contrastive loss.")

    parser.add_argument("--generation_conf", type=str, default=default_conf_file,  # required=True,
                        help="File with the configuration specifications of the models to use,"
                             " for the vector store embeddings. The columns are semicolon separated with following headers: "
                             "model_name_or_path;dataset_path;save_path;query_field;doc_field;max_length;batch_size;learning_rate;epochs")


    args = parser.parse_args()
    conf_df = pd.read_csv(args.generation_conf, sep=SEPARATOR)

    classifier_model = 'facebook/bart-large-mnli'
    summarizer_model = 'facebook/bart-base'

    generating_all_synthesis(conf_df=conf_df,
                             classifier_model=classifier_model,
                             summarizer_model=summarizer_model)





