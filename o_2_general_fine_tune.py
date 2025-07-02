"""
   This program will make run a fine-tuning of all files specified in the config file and will save
   them for further use. If you do not want to use this as a script and instead import it, then import
   only the train_dense_biencoder() function.
   WARNING: Only huggingface models are allowed.

   This program must be called using a config file in semicolon separated format. The name of
   the file must be provided as an arg to the function. The columns of such a file must be the
   following:

   - model_name_or_path :  HuggingFace model name or path to local model.
   - training_data_file : Path to CSV/TSV file containing query-document pairs.
   - finetuned_model_path : Directory to save the fine-tuned model.
   - query_field : Column name for the query in the dataset (default="query")
   - doc_column : Column name for the document in the dataset (default="document")
   - max_length : Maximum sequence length. (default=128)
   - batch_size : Batch size for training. (default=16)
   - learning_rate : Learning rate. (default=2e-5)
   - epochs : Number of training epochs. ( default=3)

   So the header of the config file MUST look like this:
   model_name_or_path;training_data_file;finetuned_model_path;query_field;doc_column;max_length;batch_size;learning_rate;epochs


   """



import os
import torch
import pandas as pd
import argparse
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from tqdm import tqdm

SEPARATOR=';'

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


class BiEncoderDataset(Dataset):
    def __init__(self, data, tokenizer, text_fields, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.query_field, self.doc_column = text_fields
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = item[self.query_field]
        doc = item[self.doc_column]

        q_tok = self.tokenizer(query, truncation=True, padding="max_length",
                               max_length=self.max_length, return_tensors="pt")
        d_tok = self.tokenizer(doc, truncation=True, padding="max_length",
                               max_length=self.max_length, return_tensors="pt")

        return {
            "query_input_ids": q_tok.input_ids.squeeze(0),
            "query_attention_mask": q_tok.attention_mask.squeeze(0),
            "doc_input_ids": d_tok.input_ids.squeeze(0),
            "doc_attention_mask": d_tok.attention_mask.squeeze(0),
        }


def mean_pooling(output, attention_mask):
    token_embeddings = output.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, dim=1) / torch.clamp(mask_expanded.sum(1), min=1e-9)


def contrastive_loss(query_emb, doc_emb, temperature=0.05):
    logits = torch.matmul(query_emb, doc_emb.T) / temperature
    labels = torch.arange(logits.size(0), device=DEVICE)#.to(query_emb.device)
    return nn.CrossEntropyLoss()(logits, labels)


def select_model(model_name:str):
    if "t5" in model_name.lower():
        from transformers import T5Tokenizer, T5EncoderModel
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5EncoderModel.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    return model, tokenizer


def train_dense_biencoder(args):

    #device = torch.device(DEVICE)

    print(f"Loading model: {args.model_name_or_path}")
    model, tokenizer = select_model(model_name=args.model_name_or_path)
    model = model.to(DEVICE)

    print(f"Loading dataset from: {args.training_data_file}")
    data = pd.read_csv(args.training_data_file, sep=SEPARATOR)
    data = data.dropna()
    data = data.to_dict(orient="records")
    dataset = BiEncoderDataset(data, tokenizer, text_fields=(args.query_field, args.doc_column), max_length=args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    model.train()
    os.makedirs(args.finetuned_model_path, exist_ok=True)

    loss_by_epoch = []
    for epoch in range(args.epochs):
        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()
            q_ids = batch["query_input_ids"].to(DEVICE)
            q_mask = batch["query_attention_mask"].to(DEVICE)
            d_ids = batch["doc_input_ids"].to(DEVICE)
            d_mask = batch["doc_attention_mask"].to(DEVICE)

            q_out = model(input_ids=q_ids, attention_mask=q_mask)
            d_out = model(input_ids=d_ids,attention_mask=d_mask)

            q_emb = mean_pooling(q_out, q_mask)
            d_emb = mean_pooling(d_out, d_mask)

            loss = contrastive_loss(q_emb, d_emb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_by_epoch.append([epoch, total_loss])

        print(f"Epoch {epoch + 1} - Loss: {total_loss:.4f}")

        model.save_pretrained(args.finetuned_model_path)
        tokenizer.save_pretrained(args.finetuned_model_path)

    loss_by_epoch_df = pd.DataFrame(loss_by_epoch, columns=['epoch', 'loss'])
    loss_by_epoch_df.to_csv(os.path.join(args.finetuned_model_path, 'loss_by_epoch.csv'), sep=SEPARATOR, index=False)
    print(f"Model saved to {args.finetuned_model_path}")

if __name__ == "__main__":
    """
    This program must be called using a config file in semicolon separated format. The name of 
    the file must be provided as an arg to the function. The columns of such a file must be the
    following:
    
    - model_name_or_path :  HuggingFace model name or path to local model.
    - training_data_file : Path to CSV/TSV file containing query-document pairs.
    - tuned_model_path : Directory to save the fine-tuned model.
    - query_field : Column name for the query in the dataset (default="query")
    - doc_column : Column name for the document in the dataset (default="document")
    - max_length : Maximum sequence length. (default=128)
    - batch_size : Batch size for training. (default=16)
    - learning_rate : Learning rate. (default=2e-5)
    - epochs : Number of training epochs. ( default=3)
    
    So the header of the config file MUST look like this:
    model_name_or_path;training_data_file;finetuned_model_path;query_field;doc_column;max_length;batch_size;learning_rate;epochs
    """

    default_conf_file = "/home/tzolkin/DebConf_2025/FineTune_data/fine_tune_DebConf2025_conf.csv"
    parser = argparse.ArgumentParser(description="Fine-tune a bi-encoder model using contrastive loss.")

    parser.add_argument("--training_conf", type=str,  default=default_conf_file, # required=True,
                            help="File with the configuration specifications of the models to fine-tune,"
                                 " columns are semicolon separated with following headers: "
                                 "model_name_or_path;training_data_file;finetuned_model_path;query_field;doc_column;max_length;batch_size;learning_rate;epochs")

    args = parser.parse_args()

    config_path = args.training_conf  # Change this to your actual config file path
    df = pd.read_csv(config_path, sep=SEPARATOR)
    del args

    for i, row in df.iterrows():
        arg_dict = row.to_dict()
        print(f" Fine tuning model {row.model_name_or_path}")

        arg_dict["model_name_or_path"] = str(arg_dict.get("model_name_or_path"))
        arg_dict["training_data_file"] = str(arg_dict.get("training_data_file"))
        arg_dict["finetuned_model_path"] = str(arg_dict.get("finetuned_model_path"))
        arg_dict["query_field"] = str(arg_dict.get("query_field"))
        arg_dict["doc_column"] = str(arg_dict.get("doc_column"))
        arg_dict["max_length"] = int(arg_dict.get("max_length"))
        arg_dict["batch_size"] = int(arg_dict.get("batch_size"))
        arg_dict["learning_rate"] = float(arg_dict.get("learning_rate"))
        arg_dict["epochs"] = int(arg_dict.get("epochs"))

        runtime_args = argparse.Namespace(**arg_dict)
        try:
            train_dense_biencoder(args=runtime_args)
        except Exception as e:
            print(f' *** ERROR while finetuning model: {arg_dict["model_name_or_path"]} : {e}')
            continue