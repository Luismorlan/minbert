from abc import ABC, abstractmethod

import torch
from typing import Any, List
from dataclasses import dataclass
import torch.nn.functional as F
import numpy as np
from bert import BertModel
from torch import nn
from tqdm import tqdm
import utils

from loss import get_perturb_loss, get_bregmman_loss, get_perturb_loss_for_pair, get_bregmman_loss_for_pair

TQDM_DISABLE = False

# Task is an abstraction for the training and evaluation of a model on a dataset.
# It itself is a wrapper on top of the baseline model with a thin layer to adapt to its own
# task specific objective.



@dataclass
class EvalResult:
    pred: List[Any]
    ids: List[Any]
    labels: List[Any]
    metric: Any
    description: str


class Task(ABC, nn.Module):
    def __init__(self, name: str, model: BertModel, train_dataloader, dev_dataloader) -> None:
        super().__init__()

        self.name = name

        # All task must share the same base BertModel.
        self.model = model

        # Dataloaders for training and evaluation.
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader

    @abstractmethod
    def loss(self, batch, pred):
        """Calculate the loss with ground truth and prediction."""
        ...

    @abstractmethod
    def evaluate(self, dataloader, is_hidden) -> EvalResult:
        ...

    @abstractmethod
    def perturbed_loss(self, *args):
        ...

    @abstractmethod
    def bregmman_loss(self, *args):
        ...


class SentimentClassificationTask(Task):
    def __init__(self, *, hidden_size: int, num_labels: int, model: BertModel, name: str, train_dataloader, dev_dataloader):
        super().__init__(name, model, train_dataloader, dev_dataloader)

        # Linear classification.
        self.proj = nn.Linear(hidden_size, num_labels)

    def forward(self, batch):
        b_ids, b_mask = (batch['token_ids'], batch['attention_mask'])
        b_embd = self.model.embed(b_ids)

        return self.forward_with_embedding(b_embd, b_mask)

    def forward_with_embedding(self, b_embd, b_mask):
        res = self.model(b_embd, b_mask)
        pooler_output = res['pooler_output']

        # (B, k)
        return self.proj(pooler_output)

    def loss(self, batch, b_pred):
        b_labels = batch['labels']
        return F.cross_entropy(b_pred, b_labels.view(-1), reduction='mean')

    def evaluate(self, dataloader, is_hidden=False) -> EvalResult:
        self.eval()
        device = next(self.model.parameters()).device

        results = {
            "pred": [],
            "ids": [],
            "labels": [],
            "metric": None,
            "description": ""
        }

        sentiment_accuracy = None

        with torch.no_grad():
            for _, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
                batch = utils.move_batch(batch, device)

                # Make prediction.
                logits = self.forward(batch)

                y_hat = logits.argmax(dim=-1).flatten().cpu().numpy()
                results["pred"].extend(y_hat)
                results["ids"].extend(batch['sent_ids'])

                if not is_hidden:
                    results["labels"].extend(
                        batch['labels'].flatten().cpu().numpy())

            if not is_hidden:
                sentiment_accuracy = np.mean(
                    np.array(results['pred']) == np.array(results['labels']))

                print(f"    sentiment acc :: {sentiment_accuracy :.3f}")
        return EvalResult(
            pred=results["pred"],
            ids=results["ids"],
            labels=results["labels"],
            metric=sentiment_accuracy,
            description="classificaiton accuracy"
        )

    def perturbed_loss(self, batch):
        return get_perturb_loss(self, batch['token_ids'], batch['attention_mask'], batch['labels'], self.args)

    def bregmman_loss(self, batch, task_tilde: nn.Module):
        return get_bregmman_loss(task_tilde, batch['token_ids'], batch['attention_mask'], batch['labels'], self.args)


# class ParaphraseDetectionTask(Task):
#     def loss(self, model, batch, device):
#         b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'],
#                                                           batch['attention_mask_1'], batch['token_ids_2'],
#                                                           batch['attention_mask_2'], batch['labels'])

#         b_ids_1 = b_ids_1.to(device)
#         b_mask_1 = b_mask_1.to(device)
#         b_ids_2 = b_ids_2.to(device)
#         b_mask_2 = b_mask_2.to(device)
#         b_labels = b_labels.to(device)

#         logits = model.predict_paraphrase(
#             b_ids_1, b_mask_1, b_ids_2, b_mask_2)
#         loss = F.cross_entropy(
#             logits, b_labels.view(-1), reduction='sum') / self.args.batch_size

#         return loss

#     def eval(self, model, dataloader, device):
#         with torch.no_grad():
#             para_y_true = []
#             para_y_pred = []
#             para_sent_ids = []
#             for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
#                 (b_ids1, b_mask1,
#                  b_ids2, b_mask2,
#                  b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
#                                           batch['token_ids_2'], batch['attention_mask_2'],
#                                           batch['labels'], batch['sent_ids'])

#                 b_ids1 = b_ids1.to(device)
#                 b_mask1 = b_mask1.to(device)
#                 b_ids2 = b_ids2.to(device)
#                 b_mask2 = b_mask2.to(device)

#                 logits = model.predict_paraphrase(
#                     b_ids1, b_mask1, b_ids2, b_mask2)
#                 # pick the most confident prediction
#                 y_hat = logits.argmax(dim=-1).flatten().cpu().numpy()
#                 b_labels = b_labels.flatten().cpu().numpy()

#                 para_y_pred.extend(y_hat)
#                 para_y_true.extend(b_labels)
#                 para_sent_ids.extend(b_sent_ids)

#             paraphrase_accuracy = np.mean(
#                 np.array(para_y_pred) == np.array(para_y_true))
#             print(f"    paraphrase acc :: {paraphrase_accuracy :.3f}")
#         return paraphrase_accuracy

#     def perturbed_loss(self, model, batch, device):
#         return get_perturb_loss_for_pair(model, batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], batch['labels'], self.args, device, 'predict_paraphrase')

#     def bregmman_loss(self, model, batch, device):
#         return get_bregmman_loss_for_pair(model, batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], batch['labels'], self.args, device, 'predict_paraphrase')


# class SemanticTextualSimilarityTask(Task):
#     def loss(self, model, batch, device):
#         b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'],
#                                                           batch['attention_mask_1'], batch['token_ids_2'],
#                                                           batch['attention_mask_2'], batch['labels'])

#         b_ids_1 = b_ids_1.to(device)
#         b_mask_1 = b_mask_1.to(device)
#         b_ids_2 = b_ids_2.to(device)
#         b_mask_2 = b_mask_2.to(device)
#         b_labels = b_labels.to(device)

#         logits = model.predict_similarity(
#             b_ids_1, b_mask_1, b_ids_2, b_mask_2)
#         loss = F.mse_loss(logits.view(-1), b_labels.view(-1),
#                           reduction='sum') / self.args.batch_size

#         return loss

#     def eval(self, model, dataloader, device):
#         with torch.no_grad():
#             sts_y_true = []
#             sts_y_pred = []
#             sts_sent_ids = []
#             for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
#                 (b_ids1, b_mask1,
#                  b_ids2, b_mask2,
#                  b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
#                                           batch['token_ids_2'], batch['attention_mask_2'],
#                                           batch['labels'], batch['sent_ids'])

#                 b_ids1 = b_ids1.to(device)
#                 b_mask1 = b_mask1.to(device)
#                 b_ids2 = b_ids2.to(device)
#                 b_mask2 = b_mask2.to(device)

#                 logits = model.predict_similarity(
#                     b_ids1, b_mask1, b_ids2, b_mask2)
#                 y_hat = logits.flatten().cpu().numpy()
#                 b_labels = b_labels.flatten().cpu().numpy()

#                 sts_y_pred.extend(y_hat)
#                 sts_y_true.extend(b_labels)
#                 sts_sent_ids.extend(b_sent_ids)

#             sts_corr = np.corrcoef(sts_y_true, sts_y_pred)[0, 1]

#         print(f"    similarity corr :: {sts_corr :.3f}")
#         return sts_corr

#     def perturbed_loss(self, model, batch, device):
#         return get_perturb_loss_for_pair(model, batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], batch['labels'], self.args, device, 'predict_similarity')

#     def bregmman_loss(self, model, batch, device):
#         return get_bregmman_loss_for_pair(model, batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], batch['labels'], self.args, device, 'predict_similarity')
