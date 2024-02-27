from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm

from loss import get_perturb_loss, get_bregmman_loss, get_perturb_loss_for_pair, get_bregmman_loss_for_pair, update_model_tilde

TQDM_DISABLE = False

# Task is an abstraction for the training and evaluation of a model on a dataset.


class Task(ABC):
    def __init__(self, args, train_dataloader, dev_dataloader) -> None:
        super().__init__()
        self.args = args
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader

    @abstractmethod
    def loss(self, model, batch, device):
        ...

    @abstractmethod
    def eval(self, model, dataloader, device):
        ...

    @abstractmethod
    def perturbed_loss(self, *args):
        ...

    @abstractmethod
    def bregmman_loss(self, *args):
        ...


class SentimentClassificationTask(Task):
    def loss(self, model, batch, device):
        b_ids, b_mask, b_labels = (batch['token_ids'],
                                   batch['attention_mask'], batch['labels'])

        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)
        b_labels = b_labels.to(device)

        logits = model.predict_sentiment(b_ids, b_mask)
        loss = F.cross_entropy(
            logits, b_labels.view(-1), reduction='sum') / self.args.batch_size

        if self.args.smart:
            loss += get_perturb_loss(model, b_ids,
                                     b_mask, logits, self.args, device)

        return loss

    def eval(self, model, dataloader, device):
        with torch.no_grad():
            sst_y_true = []
            sst_y_pred = []
            sst_sent_ids = []
            for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
                b_ids, b_mask, b_labels, b_sent_ids = batch['token_ids'], batch[
                    'attention_mask'], batch['labels'], batch['sent_ids']

                b_ids = b_ids.to(device)
                b_mask = b_mask.to(device)

                logits = model.predict_sentiment(b_ids, b_mask)
                y_hat = logits.argmax(dim=-1).flatten().cpu().numpy()
                b_labels = b_labels.flatten().cpu().numpy()

                sst_y_pred.extend(y_hat)
                sst_y_true.extend(b_labels)
                sst_sent_ids.extend(b_sent_ids)

            sentiment_accuracy = np.mean(
                np.array(sst_y_pred) == np.array(sst_y_true))

            print(f"    sentiment acc :: {sentiment_accuracy :.3f}")
        return sentiment_accuracy

    def perturbed_loss(self, model, batch, device):
        return get_perturb_loss(model, batch['token_ids'], batch['attention_mask'], batch['labels'], self.args, device, 'predict_sentiment')

    def bregmman_loss(self, model, batch, device):
        return get_bregmman_loss(model, batch['token_ids'], batch['attention_mask'], batch['labels'], self.args, device, 'predict_sentiment')


class ParaphraseDetectionTask(Task):
    def loss(self, model, batch, device):
        b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'],
                                                          batch['attention_mask_1'], batch['token_ids_2'],
                                                          batch['attention_mask_2'], batch['labels'])

        b_ids_1 = b_ids_1.to(device)
        b_mask_1 = b_mask_1.to(device)
        b_ids_2 = b_ids_2.to(device)
        b_mask_2 = b_mask_2.to(device)
        b_labels = b_labels.to(device)

        logits = model.predict_paraphrase(
            b_ids_1, b_mask_1, b_ids_2, b_mask_2)
        loss = F.cross_entropy(
            logits, b_labels.view(-1), reduction='sum') / self.args.batch_size

        return loss

    def eval(self, model, dataloader, device):
        with torch.no_grad():
            para_y_true = []
            para_y_pred = []
            para_sent_ids = []
            for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
                (b_ids1, b_mask1,
                 b_ids2, b_mask2,
                 b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                                          batch['token_ids_2'], batch['attention_mask_2'],
                                          batch['labels'], batch['sent_ids'])

                b_ids1 = b_ids1.to(device)
                b_mask1 = b_mask1.to(device)
                b_ids2 = b_ids2.to(device)
                b_mask2 = b_mask2.to(device)

                logits = model.predict_paraphrase(
                    b_ids1, b_mask1, b_ids2, b_mask2)
                y_hat = logits.sigmoid().round().flatten().cpu().numpy()
                b_labels = b_labels.flatten().cpu().numpy()

                para_y_pred.extend(y_hat)
                para_y_true.extend(b_labels)
                para_sent_ids.extend(b_sent_ids)

            paraphrase_accuracy = np.mean(
                np.array(para_y_pred) == np.array(para_y_true))
            print(f"    paraphrase acc :: {paraphrase_accuracy :.3f}")
        return paraphrase_accuracy

    def perturbed_loss(self, model, batch, device):
        return get_perturb_loss_for_pair(model, batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], batch['labels'], self.args, device, 'predict_paraphrase')

    def bregmman_loss(self, model, batch, device):
        return get_bregmman_loss_for_pair(model, batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], batch['labels'], self.args, device, 'predict_paraphrase')


class SemanticTextualSimilarityTask(Task):
    def loss(self, model, batch, device):
        b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'],
                                                          batch['attention_mask_1'], batch['token_ids_2'],
                                                          batch['attention_mask_2'], batch['labels'])

        b_ids_1 = b_ids_1.to(device)
        b_mask_1 = b_mask_1.to(device)
        b_ids_2 = b_ids_2.to(device)
        b_mask_2 = b_mask_2.to(device)
        b_labels = b_labels.to(device)

        logits = model.predict_similarity(
            b_ids_1, b_mask_1, b_ids_2, b_mask_2)
        loss = F.mse_loss(logits.view(-1), b_labels.view(-1),
                          reduction='sum') / self.args.batch_size

        return loss

    def eval(self, model, dataloader, device):
        with torch.no_grad():
            sts_y_true = []
            sts_y_pred = []
            sts_sent_ids = []
            for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
                (b_ids1, b_mask1,
                 b_ids2, b_mask2,
                 b_labels, b_sent_ids) = (batch['token_ids_1'], batch['attention_mask_1'],
                                          batch['token_ids_2'], batch['attention_mask_2'],
                                          batch['labels'], batch['sent_ids'])

                b_ids1 = b_ids1.to(device)
                b_mask1 = b_mask1.to(device)
                b_ids2 = b_ids2.to(device)
                b_mask2 = b_mask2.to(device)

                logits = model.predict_similarity(
                    b_ids1, b_mask1, b_ids2, b_mask2)
                y_hat = logits.flatten().cpu().numpy()
                b_labels = b_labels.flatten().cpu().numpy()

                sts_y_pred.extend(y_hat)
                sts_y_true.extend(b_labels)
                sts_sent_ids.extend(b_sent_ids)

            sts_corr = np.corrcoef(sts_y_true, sts_y_pred)[0, 1]

        print(f"    similarity corr :: {sts_corr :.3f}")
        return sts_corr

    def perturbed_loss(self, model, batch, device):
        return get_perturb_loss_for_pair(model, batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], batch['labels'], self.args, device, 'predict_similarity')

    def bregmman_loss(self, model, batch, device):
        return get_bregmman_loss_for_pair(model, batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], batch['labels'], self.args, device, 'predict_similarity')
