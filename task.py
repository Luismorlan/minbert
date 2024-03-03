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

from loss import get_perturb_loss, get_bregmman_loss, get_perturb_loss_for_pair

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
    def __init__(self, name: str, model: BertModel, train_dataloader, dev_dataloader, args) -> None:
        super().__init__()

        self.name = name
        self.args = args

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
    def __init__(self, *, hidden_size: int, num_labels: int, model: BertModel, name: str, train_dataloader, dev_dataloader, args):
        super().__init__(name, model, train_dataloader, dev_dataloader, args)

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

                print(f"    classification acc :: {sentiment_accuracy :.3f}")
        return EvalResult(
            pred=results["pred"],
            ids=results["ids"],
            labels=results["labels"],
            metric=sentiment_accuracy,
            description="classificaiton accuracy"
        )

    def perturbed_loss(self, batch, logits: torch.Tensor):
        return get_perturb_loss(self, batch['token_ids'], batch['attention_mask'], logits, self.args, ls_type="classifier")

    def bregmman_loss(self, batch, logits: torch.Tensor, task_tilde: nn.Module):
        return get_bregmman_loss(task_tilde, batch, logits, ls_type="classifier")


class ParaphraseDetectionTask(Task):
    def __init__(self, *, hidden_size: int, model: BertModel, name: str, train_dataloader, dev_dataloader, args):
        super().__init__(name, model, train_dataloader, dev_dataloader, args)

        # Linear layer for paraphrase detection, use 2 classes (paraphrase or not) to allow for more nuanced predictions
        self.proj = nn.Linear(2 * hidden_size, 2)

    def forward(self, batch):
        b_ids_1, b_mask_1, b_ids_2, b_mask_2 = (batch['token_ids_1'],
                                                batch['attention_mask_1'], batch['token_ids_2'],
                                                batch['attention_mask_2'])
        left = self.model.embed(b_ids_1)
        right = self.model.embed(b_ids_2)

        return self.forward_with_embeddings(left, b_mask_1, right, b_mask_2)

    def forward_with_embeddings(self, b_embd_1, b_mask_1, b_embd_2, b_mask_2):
        left = self.model(b_embd_1, b_mask_1)['pooler_output']
        right = self.model(b_embd_2, b_mask_2)['pooler_output']

        # (B, 2C)
        combined = torch.cat([left, right], dim=1)
        return self.proj(combined)

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

                print(f"    classification acc :: {sentiment_accuracy :.3f}")

        return EvalResult(
            pred=results["pred"],
            ids=results["ids"],
            labels=results["labels"],
            metric=sentiment_accuracy,
            description="classificaiton accuracy"
        )

    def perturbed_loss(self, batch, original_logits: torch.Tensor):
        return get_perturb_loss_for_pair(self, batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], original_logits, self.args, ls_type="classifier")

    def bregmman_loss(self, batch, logits: torch.Tensor, task_tilde: nn.Module):
        return get_bregmman_loss(task_tilde, batch, logits, ls_type="classifier")


class SemanticTextualSimilarityTask(Task):
    def __init__(self, *, hidden_size: int, model: BertModel, name: str, train_dataloader, dev_dataloader, args):
        super().__init__(name, model, train_dataloader, dev_dataloader, args)

        # Linear layer for similarity detection.
        self.proj = nn.Linear(2 * hidden_size, 1)

    def forward(self, batch):
        b_ids_1, b_mask_1, b_ids_2, b_mask_2 = (batch['token_ids_1'],
                                                batch['attention_mask_1'], batch['token_ids_2'],
                                                batch['attention_mask_2'])
        left = self.model.embed(b_ids_1)
        right = self.model.embed(b_ids_2)

        return self.forward_with_embedding(left, b_mask_1, right, b_mask_2)

    def forward_with_embedding(self, b_embd_1, b_mask_1, b_embd_2, b_mask_2):
        left = self.model(b_embd_1, b_mask_1)['pooler_output']
        right = self.model(b_embd_2, b_mask_2)['pooler_output']

        # (B, 2C)
        combined = torch.cat([left, right], dim=1)

        # (B, 1), map values to [0, 5]
        return 5 * torch.sigmoid(self.proj(combined))

    def loss(self, batch, b_pred):
        b_labels = batch['labels']
        return F.mse_loss(b_pred.view(-1), b_labels.view(-1), reduction='mean')

    def evaluate(self, dataloader, is_hidden=False) -> EvalResult:
        self.eval()
        device = next(self.model.parameters()).device

        results = {
            "pred": [],
            "ids": [],
            "labels": [],
            "metric": None,
        }

        sts_corr = None

        with torch.no_grad():
            for _, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
                batch = utils.move_batch(batch, device)

                # Make prediction. This is directly a number between 0-5.
                logits = self.forward(batch)

                y_hat = logits.flatten().cpu().numpy()
                results["pred"].extend(y_hat)
                results["ids"].extend(batch['sent_ids'])

                if not is_hidden:
                    results["labels"].extend(
                        batch['labels'].flatten().cpu().numpy())

            if not is_hidden:
                sts_corr = np.corrcoef(
                    results['labels'], results['pred'])[0, 1]
                print(f"    standard correlation :: {sts_corr :.3f}")

        return EvalResult(
            pred=results["pred"],
            ids=results["ids"],
            labels=results["labels"],
            metric=sts_corr,
            description="correlation with true label"
        )

    def perturbed_loss(self, batch, original_logits: torch.Tensor):
        return get_perturb_loss_for_pair(self, batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], original_logits, self.args, ls_type="regressor")

    def bregmman_loss(self, batch, logits: torch.Tensor, task_tilde: nn.Module):
        return get_bregmman_loss(task_tilde, batch, logits, ls_type="regressor")
