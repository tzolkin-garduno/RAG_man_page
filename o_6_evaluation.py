import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.metrics import roc_curve, auc

from o_4_retrieve_documents import remove_dir

SEPARATOR = ';'

def evaluate_retriever(ground_truth_path: str, rag_results_path: str,
                       evaluation_score_col: str = 'positive_score', score_order: str = 'ascending'):
    # Load data
    rag_results = pd.read_csv(rag_results_path, sep=SEPARATOR)
    ground_truth = pd.read_csv(ground_truth_path, sep=SEPARATOR)
    ground_truth['qa_type'] = ground_truth['qa_type'].apply(lambda t : 'negative' if 'negative' in t else t)

    # Merge on chunk_num, source_file and query
    merged = pd.merge(
        rag_results,
        ground_truth,
        #on=["chunk_num", "source_file", "query"],
        on=["source_file", "query"],
        suffixes=("_rag", "_gt")
    )

    # Ground truth positive label
    merged["is_true_positive"] = merged["qa_type_gt"] == "positive"

    # Create thresholds if it is an ascending score then, that is , greater means more relevant, then start from beneath
    # If it is a descending score
    if score_order== 'ascending':
        merged = merged.sort_values(by=evaluation_score_col, ascending=True)
        thresholds = np.linspace(merged[evaluation_score_col].min(), merged[evaluation_score_col].max(), 100)
    else:
        merged = merged.sort_values(by=evaluation_score_col, ascending=False)
        thresholds = np.linspace(merged[evaluation_score_col].max(), merged[evaluation_score_col].min(), 100)

    precision = []
    recall = []

    for thresh in thresholds:
        if score_order== 'ascending':
            selected = merged[merged[evaluation_score_col] >= thresh]
        else:
            selected = merged[merged[evaluation_score_col] <= thresh]


        if len(selected) == 0:
            precision.append(0)
            recall.append(0)
            continue

        tp = selected["is_true_positive"].sum()
        fp = len(selected) - tp
        fn = merged["is_true_positive"].sum() - tp

        precision.append(tp / (tp + fp) if (tp + fp) > 0 else 0)
        recall.append(tp / (tp + fn) if (tp + fn) > 0 else 0)

    return thresholds, precision, recall, merged


def plot_roc_curve(merged_df: pd.DataFrame,
                   score_column: str,
                   model_name: str,
                   save_2_file: str = None,
                   display: bool = False,
                   score_order: str = 'ascending'):
    y_true = merged_df["is_true_positive"].astype(int)
    if score_order == 'ascending':
        y_scores = merged_df[score_column]  # Higher score means more relevant
    else:
        y_scores = -merged_df[score_column] # Lower score means more relevant

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC Curve ")
    plt.legend(loc="lower right")
    #plt.grid(True)
    plt.tight_layout()
    if save_2_file:
        plt.savefig(save_2_file, format=save_2_file.split('.')[1],  dpi=300)
    if display:
        plt.show()
    else:
        plt.close()
    print(f"AUC: {roc_auc:.4f}")

    return tpr, fpr, roc_auc


def plot_all_roc_curves(df: pd.DataFrame, display: bool = False, save_2_file: str = None):
    plt.figure(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(df)))  # Generate distinct colors

    for (idx, row), color in zip(df.iterrows(), colors):
        plt.plot(row.fpr, row.tpr, color=color, lw=2,
                 label=f"MODEL={row.model_name} AUC = {row.auc:.2f}")

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("All ROC Curves")
    plt.legend(loc="lower right")
    #plt.grid(True)
    plt.tight_layout()

    if save_2_file:
        plt.savefig(save_2_file, format=save_2_file.split('.')[-1], dpi=300)
    if display:
        plt.show()
    else:
        plt.close()


def display_precision_recall(thresholds, precision, recall, model_name:str):
    plt.figure(figsize=(10, 5))
    plt.plot(thresholds, precision, label="Precision")
    plt.plot(thresholds, recall, label="Recall")
    plt.xlabel("Positive Score Threshold")
    plt.ylabel("Score")
    plt.title(f"{model_name}Precision and Recall vs. Positive Score Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print()


def evaluation_results(conf_df: pd.DataFrame):

    all_results = []
    for i, row in conf_df.iterrows():
        print(f"📁 Loading generated responses docs from: {row.generated_responses_file}")
        remove_dir(dir_path=row.evaluation_path)
        os.makedirs(row.evaluation_path, exist_ok=True)

        thresholds, precision, recall, merged_df = evaluate_retriever(ground_truth_path=row.ground_truth_data_file,
                                                                      rag_results_path=row.generated_responses_file,
                                                                      evaluation_score_col='distance',
                                                                      score_order='descending')

        results_table = [[t, p, r] for t, p, r in  zip(thresholds, precision, recall)]
        results_df = pd.DataFrame(results_table, columns = ['threshold', 'precision', 'recall'])
        results_df.to_csv(os.path.join(row.evaluation_path, 'evaluation_results.csv'), sep=SEPARATOR, index=False)
        #display_precision_recall(thresholds, precision, recall, row.model_name_or_path.split('/')[1])

        #plot_roc_curve(merged_df=merged_df, score_column='positive_score', score_order='ascending')

        tpr, fpr, auc = plot_roc_curve(merged_df=merged_df, score_column='distance', score_order='descending',
                       model_name=row.model_name_or_path.split('/')[1],
                       display=False, save_2_file=os.path.join(row.evaluation_path, 'roc_curve.png'))
        all_results.append([row.model_name_or_path.split('/')[1], tpr, fpr, auc])

    all_results = pd.DataFrame(all_results, columns=['model_name', 'tpr', 'fpr', 'auc'])

    evaluation_path = os.path.join(os.path.dirname(conf_df.loc[0,'evaluation_path']), 'all_roc_curves.png')
    plot_all_roc_curves(df=all_results, display=True, save_2_file= evaluation_path)


def main():
    # Load data
    default_conf_file = "/home/tzolkin/DebConf_2025/FineTune_data/fine_tune_DebConf2025_conf.csv"
    parser = argparse.ArgumentParser(description="Fine-tune a bi-encoder model using contrastive loss.")

    parser.add_argument("--evaluation_conf", type=str, default=default_conf_file,  # required=True,
                        help="File with the configuration specifications of the evaluation."
                             "The columns are semicolon separated with following headers: "
                             "model_name_or_path;dataset_path;save_path;query_field;doc_field;max_length;batch_size;learning_rate;epochs")

    args = parser.parse_args()
    conf_df = pd.read_csv(args.evaluation_conf, sep=SEPARATOR)
    evaluation_results(conf_df=conf_df)


if __name__ == "__main__":
    main()
