#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author:fsdjo
@file: finetune.py
@time: 2022/01/12
"""
import transformers
import torch, random
from tqdm import tqdm
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from model import BertWithKeywordForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
import numpy as np

transformers.logging.set_verbosity_error()


def load_data(train_csv, test_csv):
    train_data = pd.read_csv(train_csv, header=None, sep="\t", error_bad_lines=False)
    test_data = pd.read_csv(test_csv, header=None, sep="\t", error_bad_lines=False)
    return train_data, test_data


def convert_to_dataset_torch(data):
    input_ids = []
    attention_masks = []
    token_type_ids = []
    kw_attention_masks = []
    real_text_a_masks = []
    real_text_b_masks = []
    sents1, sents2, labels,sents1_keywords,sents2_keywords = [], [], [],[],[]

    for sent1, sent2, label,sent1_keyword,sent2_keyword in zip(data[0].tolist(), data[1].tolist(), data[2].tolist(),data[3].tolist(),data[4].tolist()):
        if label not in [1.0, 0.0]:
            continue
        sents1.append(str(sent1))
        sents2.append(str(sent2))
        if str(sent1_keyword) == "nan":
            sents1_keywords.append([])
        else:
            sents1_keywords.append(str(sent1_keyword).split("|"))
        if str(sent2_keyword) == "nan":
            sents2_keywords.append([])
        else:
            sents2_keywords.append(str(sent2_keyword).split("|"))
        labels.append(int(label))

    for sent1, sent2,sent1_keyword,sent2_keyword in tqdm(zip(sents1, sents2,sents1_keywords,sents2_keywords), total=data.shape[0]):
        encoded_dict = tokenizer.encode_plus(sent1, sent2, max_length=max_length,
                                             pad_to_max_length=True,
                                             return_attention_mask=True, truncation=True)
        input_ids.append(torch.tensor(encoded_dict['input_ids']).unsqueeze(0))
        token_type_ids.append(torch.tensor(encoded_dict['token_type_ids']).unsqueeze(0))
        attention_masks.append(torch.tensor(encoded_dict['attention_mask']).unsqueeze(0))
        real_text_a_mask = [0] + [1] * (encoded_dict['input_ids'].index(102) - 1) + [0] * (
                    len(encoded_dict['input_ids']) - encoded_dict['input_ids'].index(102))
        real_text_b_mask = [0] * (encoded_dict['input_ids'].index(102) + 1) + [1] * (
                    len(encoded_dict['input_ids']) - encoded_dict['input_ids'].index(102) -2) + [0]
        real_text_a_masks.append(torch.tensor(real_text_a_mask).unsqueeze(0))
        real_text_b_masks.append(torch.tensor(real_text_b_mask).unsqueeze(0))
        kw_attention_mask = [0]*len(encoded_dict["input_ids"])
        for keyword in sent2_keyword+sent1_keyword:
            if keyword in sent1:
                kw_attention_mask[1+sent1.index(keyword):1+sent1.index(keyword)+len(keyword)]=[1]*len(keyword)
            if keyword in sent2:
                kw_attention_mask[2+len(sent1)+sent2.index(keyword):2+len(sent1)+sent2.index(keyword)+len(keyword)]=[1]*len(keyword)
        kw_attention_masks.append(torch.tensor(kw_attention_mask).unsqueeze(0))

    input_ids = torch.cat(input_ids, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    real_text_a_masks = torch.cat(real_text_a_masks,dim=0)
    real_text_b_masks = torch.cat(real_text_b_masks,dim=0)
    kw_attention_masks = torch.cat(kw_attention_masks,dim=0)
    labels = torch.tensor(labels)
    return TensorDataset(input_ids, attention_masks, token_type_ids, labels,real_text_a_masks,real_text_b_masks,kw_attention_masks)


def fit_batch(dataloader, model, optimizer, epoch):
    total_train_loss = 0

    for batch in tqdm(dataloader, desc=f"Training epoch:{epoch + 1}", unit="batch"):
        # Unpack batch from dataloader.
        input_ids = batch[0].to(device)
        attention_masks = batch[1].to(device)
        token_type_ids = batch[2].to(device)
        labels = batch[3].to(device)
        real_text_a_masks = batch[4].to(device)
        real_text_b_masks = batch[5].to(device)
        kw_attention_masks = batch[6].to(device)
        model.zero_grad()
        outputs = model(input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_masks,
                        labels=labels,
                        keyword_mask=kw_attention_masks,
                        real_mask_a=real_text_a_masks,
                        real_mask_b=real_text_b_masks)
        loss = outputs[0]
        total_train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    return total_train_loss


def eval_batch(dataloader, model, metric=accuracy_score):
    total_eval_accuracy = 0
    total_eval_loss = 0
    predictions, predicted_labels = [], []

    for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
        input_ids = batch[0].to(device)
        attention_masks = batch[1].to(device)
        token_type_ids = batch[2].to(device)
        labels = batch[3].to(device)
        real_text_a_masks = batch[4].to(device)                                                                                                                     
        real_text_b_masks = batch[5].to(device)                                                                                                                     
        kw_attention_masks = batch[6].to(device)
        with torch.no_grad():
            outputs = model(input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_masks,
                            labels=labels,                                                                                                
                            keyword_mask=kw_attention_masks,                                                   
                        real_mask_a=real_text_a_masks,                                                                                                              
                        real_mask_b=real_text_b_masks)
            loss = outputs[0]
            logits = outputs[1]
        total_eval_loss += loss.item()
        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        y_pred = np.argmax(logits, axis=1).flatten()
        total_eval_accuracy += metric(label_ids, y_pred)
        predictions.extend(logits.tolist())
        predicted_labels.extend(y_pred.tolist())
    return total_eval_accuracy, total_eval_loss, predictions, predicted_labels


def train_epoch(train_dataloader, validation_dataloader, model, optimizer, epochs):
    training_stats = []

    for epoch in range(0, epochs):
        model.train()
        total_train_loss = fit_batch(train_dataloader, model, optimizer, epoch)
        avg_train_loss = total_train_loss / len(train_dataloader)
        model.eval()
        total_eval_accuracy, total_eval_loss, _, _ = eval_batch(validation_dataloader, model)
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("\n")
        print(f"score: {avg_val_accuracy}")

        avg_val_loss = total_eval_loss / len(validation_dataloader)
        print(f"Validation Loss: {avg_val_loss}")
        print("\n")

        training_stats.append(
            {
                'epoch': epoch,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. score.': avg_val_accuracy
            }
        )
    print("Training complete!")
    return training_stats

if __name__=="__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cpu")
        n_gpu = 0

    max_length = 128
    model_name = "/nfs/project/similarity/pretrain/mengzi-bert-base"
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
    train_csv = "./train.csv"
    test_csv = "./test.csv"
    train_df, test_df = load_data(train_csv, test_csv)
    train = convert_to_dataset_torch(train_df)
    validation = convert_to_dataset_torch(test_df)
    batch_size = 64

    train_dataloader = DataLoader(train, sampler=RandomSampler(train), batch_size=batch_size)
    validation_dataloader = DataLoader(validation, sampler=SequentialSampler(validation), batch_size=batch_size)
    tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
    config = BertConfig.from_pretrained(model_name)
    model = BertWithKeywordForSequenceClassification.from_pretrained(model_name, num_labels=2,config=config)
    model.cuda()
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epochs = 4
    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    seed_val = 2020

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed_val)

    training_stats = train_epoch(train_dataloader, validation_dataloader, model, optimizer, epochs)

    torch.save(model.state_dict(), "./outputs/mengzi-bert-base/pytorch_model.bin")
