'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import random
import numpy as np
import argparse
from types import SimpleNamespace
from typing import Any, Callable, List
from abc import ABC, abstractmethod
from itertools import cycle
import copy

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from task import TQDM_DISABLE, Task, SentimentClassificationTask, ParaphraseDetectionTask, SemanticTextualSimilarityTask
from loss import update_model_tilde

from tqdm import tqdm

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from evaluation import model_eval_multitask, model_eval_test_multitask


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''

    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Pretrain mode does not require updating BERT paramters.
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True
        # You will want to add layers here to perform the downstream tasks.
        # Linear layer for multi-class sentiment classification
        self.sent_linear = nn.Linear(config.hidden_size, config.num_labels)
        # Linear layer for paraphrase detection, use 2 classes (paraphrase or not) to allow for more nuanced predictions
        self.para_linear = nn.Linear(2*config.hidden_size, 2)
        # Linear layer for semantic textual similarity
        self.sts_linear = nn.Linear(2*config.hidden_size, 1)

    def embed(self, input_ids):
        '''Takes a batch of sentences and returns BERT embeddings'''
        return self.bert.embed(input_ids=input_ids)

    def forward(self, ids_or_embedding, attention_mask, is_embedding=False):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        if is_embedding:
            res = self.bert.forward_with_embedding(
                ids_or_embedding, attention_mask)
        else:
            res = self.bert(ids_or_embedding, attention_mask)

        return res

    def predict_sentiment(self, ids_or_embedding, attention_mask, is_embedding=False):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        res = self.forward(ids_or_embedding, attention_mask, is_embedding=is_embedding)

        # Size: (B, C)
        pooler_output = res["pooler_output"]

        return self.sent_linear(pooler_output)

    def predict_paraphrase(self,
                           ids_or_embedding_1, attention_mask_1,
                           ids_or_embedding_2, attention_mask_2,
                           is_embedding=False):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        res1 = self.forward(ids_or_embedding_1, attention_mask_1, is_embedding=is_embedding)
        res2 = self.forward(ids_or_embedding_2, attention_mask_2, is_embedding=is_embedding)

        # Size: (B, 2*C)
        pooler_output_1 = res1["pooler_output"]
        pooler_output_2 = res2["pooler_output"]

        # Combine the two embeddings
        combined = torch.cat((pooler_output_1, pooler_output_2), dim=1)

        return self.para_linear(combined)

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''

        res1 = self.forward(input_ids_1, attention_mask_1)
        res2 = self.forward(input_ids_2, attention_mask_2)

        # Size: (B, 2*C)
        pooler_output_1 = res1["pooler_output"]
        pooler_output_2 = res2["pooler_output"]

        # Combine the two embeddings
        combined = torch.cat((pooler_output_1, pooler_output_2), dim=1)

        return self.sts_linear(combined)


class Trainer:
    def __init__(self, args: Any, config: SimpleNamespace, tasks: List[Task]) -> None:
        self.args = args
        self.config = config
        self.tasks = tasks

    def train(self, model: nn.Module):
        best_dev_metrics = [0,] * len(self.tasks) # higher the better
        acc_dev = [0,] * len(self.tasks) # higher the better
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        optimizer = AdamW(model.parameters(), lr=self.args.lr)
        model.train()

        train_dataloaders = [task.train_dataloader for task in self.tasks]
        dev_dataloaders = [task.dev_dataloader for task in self.tasks]

        max_len, longest_dl = max([(len(dl), dl) for dl in train_dataloaders], key=lambda x: x[0])
        train_dataloaders = [cycle(dl) if dl != longest_dl else dl for dl in train_dataloaders]
        model_tilde = copy.deepcopy(model) if args.smart else None

        for epoch in range(self.args.epochs):
            num_batches = 0
            train_loss = 0

            progress_bar = tqdm(range(self.args.epochs), desc=f'train-{epoch}', disable=TQDM_DISABLE, total=max_len)
            # we zip the dataloaders together to train on all tasks at the same time
            train_iter = iter(zip(*train_dataloaders))

            # import pdb; pdb.set_trace()

            while True:
                try:
                    if not args.smart:
                        batches = next(train_iter)
                        loss = 0
                        optimizer.zero_grad()
                        for batch, task in zip(batches, self.tasks):
                            loss += task.loss(model, batch, device)

                        loss.backward()
                        optimizer.step()

                        train_loss += loss.item()
                        num_batches += 1
                        progress_bar.update(1)
                    else:
                        for _ in range(args.s):
                            loss = 0
                            optimizer.zero_grad()
                            batches = next(train_iter)
                            for batch, task in zip(batches, self.tasks):
                                loss += task.loss(model, batch, device)
                                loss += args.lambda_s * task.perturbed_loss(model, batch, device)
                                loss += args.mu * task.get_bregmman_loss(model, batch, device)

                            loss.backward()
                            optimizer.step()
                            train_loss += loss.item()
                            num_batches += 1
                            progress_bar.update(1)

                        update_model_tilde(model_tilde, model, args.beta)

                except StopIteration:
                    break

            progress_bar.close()
            
            train_loss = train_loss / (num_batches)

            print(f"Epoch {epoch}:\n train loss :: {train_loss :.3f}")
            for i, (task, train_dataloader, dev_dataloader) in enumerate(zip(self.tasks, train_dataloaders, dev_dataloaders)):
                task.eval(model, train_dataloader, device)
                acc_dev[i] = task.eval(model, dev_dataloader, device)

            # TODO: how to weight these dev metrics across different tasks?
            if np.average(acc_dev) > np.average(best_dev_metrics):
                best_dev_metrics = acc_dev
                save_model(model, optimizer, self.args, self.config, self.args.filepath)


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(
        args.sst_train, args.para_train, args.sts_train, split='train')
    sst_dev_data, num_labels, para_dev_data, sts_dev_data = load_multitask_data(
        args.sst_dev, args.para_dev, args.sts_dev, split='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)
    sts_train_data = SentencePairDataset(
        sts_train_data, args, isRegression=True)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                       collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=para_dev_data.collate_fn)
    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)
    # Init tasks
    sst_task = SentimentClassificationTask(args, sst_train_dataloader, sst_dev_dataloader)
    para_task = ParaphraseDetectionTask(args, para_train_dataloader, para_dev_dataloader)
    sts_task = SemanticTextualSimilarityTask(args, sts_train_dataloader, sts_dev_dataloader)


    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    # Init Trainer
    trainer = Trainer(args, config, [sst_task, para_task, sts_task])

    # Train
    trainer.train(model)



def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels, para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test, args.para_test,
                                args.sts_test, split='test')

        sst_dev_data, num_labels, para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev, args.para_dev,
                                args.sts_dev, split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(
            sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy, dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                                  para_dev_dataloader,
                                                                                  sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
            model_eval_test_multitask(sst_test_dataloader,
                                      para_test_dataloader,
                                      sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str,
                        default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str,
                        default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str,
                        default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str,
                        default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str,
                        default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str,
                        default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str,
                        default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str,
                        default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str,
                        default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str,
                        default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str,
                        default="predictions/sts-test-output.csv")

    parser.add_argument(
        "--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument("--smart", type=bool, default=False,
                        help='use SMART (https://arxiv.org/abs/1911.03437) update to fine-tune the model')
    parser.add_argument("--lambda_s", type=float, default=1.0,
                        help='lambda_s for the SMART update, only used when --smart is True')
    parser.add_argument("--sigma", type=float, default=1e-5,
                        help='sigma for the SMART update, only used when --smart is True')
    parser.add_argument("--epsilon", type=float, default=1e-5,
                        help='epsilon for the SMART update, only used when --smart is True')
    parser.add_argument("--eta", type=float, default=1e-3,
                        help='eta for the SMART update, only used when --smart is True')
    parser.add_argument("--tx", type=int, default=1,
                        help='Iteration size of x for the SMART update, only used when --smart is True')
    parser.add_argument("--s", type=int, default=1,
                        help='Iteration size of S for the SMART update, only used when --smart is True. This is to perform update within the trust region.')
    parser.add_argument("--dataset", nargs="*", choices=[
                        'sst', 'cfimdb', 'quora', 'semeval'], default=["sst", "cfimdb"], help="List of datasets that can be used to train or finetune.")
    parser.add_argument("--mu", type=float,
                        help="Coefficient for Bregmma loss", default=1)
    # TODO: use a rate schedule for beta. In the paper it is decreasing to 0.999 after 10% of training.
    parser.add_argument(
        "--beta", type=float, help="Coefficient for momentum of theta tilde", default=0.99)
    # TODO: actually implement this
    parser.add_argument("--contrastive", type=bool, default=False,
                        help="Use contrastive loss fromm https://arxiv.org/pdf/2104.08821.pdf")
    parser.add_argument(
        "--tau", type=float, help="Tau coefficient for the contrastive loss", default=0.1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    # Save path.
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt'
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train_multitask(args)
    test_multitask(args)
