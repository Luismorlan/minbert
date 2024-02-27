from atexit import register
import random
from this import d
from typing import Any
import numpy as np
import argparse
import torch.nn as nn
from types import SimpleNamespace
import csv
import copy

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, r2_score

from tokenizer import BertTokenizer
from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm


TQDM_DISABLE = False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class BertSentimentClassifier(torch.nn.Module):
    '''
    This module performs sentiment classification using BERT embeddings on the SST dataset.

    In the SST dataset, there are 5 sentiment categories (from 0 - "negative" to 4 - "positive").
    Thus, your forward() should return one logit for each of the 5 classes.
    '''

    def __init__(self, config):
        super(BertSentimentClassifier, self).__init__()
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Pretrain mode does not require updating BERT paramters.
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True

        # Create any instance variables you need to classify the sentiment of BERT embeddings.
        self.pooler_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.proj = nn.Linear(config.hidden_size, config.num_labels)

    def embed(self, input_ids):
        '''Takes a batch of sentences and returns BERT embeddings'''
        return self.bert.embed(input_ids=input_ids)

    def forward(self, ids_or_embedding, attention_mask, is_embedding=False):
        '''Takes a batch of sentences and returns logits for sentiment classes'''
        # The final BERT contextualized embedding is the hidden state of [CLS] token (the first token).
        # HINT: You should consider what is an appropriate return value given that
        # the training loop currently uses F.cross_entropy as the loss function.
        if is_embedding:
            res = self.bert.forward_with_embedding(
                ids_or_embedding, attention_mask)
        else:
            res = self.bert(ids_or_embedding, attention_mask)

        # Size: (B, C)
        pooler_output = res["pooler_output"]

        # Size: (B, L)
        # Applying dropout achieves worse performance.
        # out = self.proj(self.pooler_dropout(pooler_output))
        out = self.proj(pooler_output)

        # Directly output the raw logits so that it can be used in the loss function.
        return out


class BertRegressor(torch.nn.Module):
    '''
    This module performs the regression task using BERT embeddings.

    This is used for tasks such as SemEval where the output is a real number.
    '''

    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Pretrain mode does not require updating BERT paramters.
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True

        # Linear projection from the feature to a real number for regression task.
        self.proj = nn.Linear(config.hidden_size, 1)

    def embed(self, input_ids):
        '''Takes a batch of sentences and returns BERT embeddings'''
        return self.bert.embed(input_ids=input_ids)

    def forward(self, ids_or_embedding, attention_mask, is_embedding=False):
        '''Takes a batch of sentences and returns logits for sentiment classes'''
        # The final BERT contextualized embedding is the hidden state of [CLS] token (the first token).
        # HINT: You should consider what is an appropriate return value given that
        # the training loop currently uses F.cross_entropy as the loss function.
        if is_embedding:
            res = self.bert.forward_with_embedding(
                ids_or_embedding, attention_mask)
        else:
            res = self.bert(ids_or_embedding, attention_mask)

        # Size: (B, C)
        pooler_output = res["pooler_output"]

        # Size: (B, 1)
        out = self.proj(pooler_output)

        # scale up to between [0, 5]
        return 5 * torch.sigmoid(out)


class BaseDataSets(Dataset):
    def __init__(self, dataset, args, is_test=False):
        self.dataset = dataset
        self.p = args
        self.is_test = is_test
        self.label_type = torch.FloatTensor if args.task_type == 'regressor' else torch.LongTensor
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class SentimentDataset(BaseDataSets):
    def pad_data(self, data):
        sents = [x[0] for x in data]
        if self.is_test:
            sent_ids = [x[1] for x in data]
        else:
            labels = [x[1] for x in data]
            sent_ids = [x[2] for x in data]

        encoding = self.tokenizer(
            sents, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])

        if self.is_test:
            return token_ids, attention_mask, sents, sent_ids
        else:
            labels = torch.LongTensor(labels)
            return token_ids, attention_mask, labels, sents, sent_ids

    def collate_fn(self, all_data):
        if self.is_test:
            token_ids, attention_mask, sents, sent_ids = self.pad_data(
                all_data)

            return {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'sents': sents,
                'sent_ids': sent_ids
            }

        else:
            token_ids, attention_mask, labels, sents, sent_ids = self.pad_data(
                all_data)

            return {
                'token_ids': token_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'sents': sents,
                'sent_ids': sent_ids
            }


class SentencePairDataset(BaseDataSets):
    def collate_fn(self, all_data):
        sents1 = [x[0] for x in all_data]
        sents2 = [x[1] for x in all_data]
        labels = [x[2] for x in all_data]
        sent_ids = [x[3] for x in all_data]

        encoding = self.tokenizer(
            sents1, sents2, return_tensors='pt', padding=True, truncation=True)
        token_ids = torch.LongTensor(encoding['input_ids'])
        token_type_ids = torch.LongTensor(encoding['token_type_ids'])
        attention_mask = torch.LongTensor(encoding['attention_mask'])
        labels = self.label_type(labels)

        return {
            'token_ids': token_ids,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'sents1': sents1,
            'sents2': sents2,
            'sent_ids': sent_ids
        }


# Load the data: a list of (sentence, label).
def load_data(args, mode='train'):
    num_labels = {}
    data = []
    filename = args.train if mode == 'train' else args.dev if mode == 'valid' else args.test
    if args.dataset == 'semeval':
        with open(filename, 'r') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent1 = record['sentence1'].lower().strip()
                sent2 = record['sentence2'].lower().strip()
                sent_id = record['id'].lower().strip()
                if mode == 'test':
                    data.append((sent1, sent2, sent_id))
                else:
                    label = float(record['similarity'].strip())
                    data.append((sent1, sent2, label, sent_id))
    elif args.dataset == 'quora':
        with open(filename, 'r') as fp:
            for record in csv.DictReader(fp, delimiter='\t'):
                sent1 = record['sentence1'].lower().strip()
                sent2 = record['sentence2'].lower().strip()
                sent_id = record['id'].lower().strip()
                if mode == 'test':
                    data.append((sent1, sent2, sent_id))
                else:
                    label = int(record['is_duplicate'].strip())
                    data.append((sent1, sent2, label, sent_id))
    else:
        if mode == 'test':
            with open(filename, 'r') as fp:
                for record in csv.DictReader(fp, delimiter='\t'):
                    sent = record['sentence'].lower().strip()
                    sent_id = record['id'].lower().strip()
                    data.append((sent, sent_id))
        else:
            with open(filename, 'r') as fp:
                for record in csv.DictReader(fp, delimiter='\t'):
                    sent = record['sentence'].lower().strip()
                    sent_id = record['id'].lower().strip()
                    label = int(record['sentiment'].strip())
                    if label not in num_labels:
                        num_labels[label] = len(num_labels)
                    data.append((sent, label, sent_id))
            print(f"load {len(data)} data from {filename}")

    if mode == 'train':
        return data, len(num_labels)
    else:
        return data, None


# Evaluate the model on dev examples.
def model_eval(dataloader, model, device, args):
    # Switch to eval model, will turn off randomness like dropout.
    model.eval()
    y_true = []
    y_pred = []
    sents = []
    sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        if args.dataset == 'semeval' or args.dataset == 'quora':
            b_ids, b_type_ids, b_mask, b_labels, b_sents1, b_sents2, b_sent_ids = batch['token_ids'], batch['token_type_ids'], batch['attention_mask'],  \
                batch['labels'], batch['sents1'], batch['sents2'], batch['sent_ids']
            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)

            output = model(b_ids, b_mask)
            preds = output.detach().cpu().numpy()

            b_labels = b_labels.flatten()
            y_true.extend(b_labels)
            y_pred.extend(preds)
            sents.extend(b_sents1)
            sents.extend(b_sents2)
            sent_ids.extend(b_sent_ids)

            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)

            return mse, r2, y_pred, y_true, sents, sent_ids

        else:
            b_ids, b_mask, b_labels, b_sents, b_sent_ids = batch['token_ids'], batch['attention_mask'],  \
                batch['labels'], batch['sents'], batch['sent_ids']

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)

            logits = model(b_ids, b_mask)
            logits = logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1).flatten()

            b_labels = b_labels.flatten()
            y_true.extend(b_labels)
            y_pred.extend(preds)
            sents.extend(b_sents)
            sent_ids.extend(b_sent_ids)

            f1 = f1_score(y_true, y_pred, average='macro')
            acc = accuracy_score(y_true, y_pred)

        return acc, f1, y_pred, y_true, sents, sent_ids


# Evaluate the model on test examples.
def model_test_eval(dataloader, model, device, args):
    # Switch to eval model, will turn off randomness like dropout.
    model.eval()
    y_pred = []
    sents = []
    sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        if args.dataset == 'semeval' or args.dataset == 'quora':
            b_ids, b_type_ids, b_mask, b_labels, b_sents1, b_sents2, b_sent_ids = batch['token_ids'], batch['token_type_ids'], batch['attention_mask'],  \
                batch['labels'], batch['sents1'], batch['sents2'], batch['sent_ids']
            b_sents = b_sents1 + b_sents2
            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)

            output = model(b_ids, b_mask)
            preds = output.detach().cpu().numpy()

        else:
            b_ids, b_mask, b_sents, b_sent_ids = batch['token_ids'], batch['attention_mask'],  \
                batch['sents'], batch['sent_ids']

            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)

            logits = model(b_ids, b_mask)
            logits = logits.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1).flatten()

        y_pred.extend(preds)
        sents.extend(b_sents)
        sent_ids.extend(b_sent_ids)

    return y_pred, sents, sent_ids


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


def l_s(p, q, type="classifier"):
    """Implementation of the L_s loss function for the SMART update."""
    if type == "classifier":
        return F.kl_div(
            F.log_softmax(p, dim=-1),
            F.log_softmax(q, dim=-1),
            reduction='batchmean',
            log_target=True,
        ) + F.kl_div(
            F.log_softmax(q, dim=-1),
            F.log_softmax(p, dim=-1),
            reduction='batchmean',
            log_target=True
        )
    elif type == "regressor":
        return F.mse_loss(p, q, reduction='mean')


def get_perturb_loss(model: nn.Module, b_ids: torch.Tensor, b_mask: torch.Tensor, orginal_logits: torch.Tensor, args: Any, device: Any):
    # In addition to the standard cross-entropy loss, we also add the SMART loss.
    # Compute the embedding of the batch
    start_embeddings = model.embed(b_ids)

    # Perturb the embedding with Gaussian noise.
    embeddings_perturbed: torch.Tensor = start_embeddings + \
        torch.normal(0, args.sigma, start_embeddings.size()).to(device)

    # Loop until tx iterations have been performed
    for _ in range(args.tx):
        # Compute the gradient of the loss with respect to the perturbed embedding.
        embeddings_perturbed.requires_grad_()
        logits = model(embeddings_perturbed, b_mask, is_embedding=True)

        # Use symmetrizied KL divergence as the loss function.
        # TODO: unify the l_s calculation into a single function that also usable for regression task.
        loss_perturbed = l_s(logits, orginal_logits, type=args.task_type)

        grad = torch.autograd.grad(
            loss_perturbed, embeddings_perturbed)[0]
        # Normalize the gradient by infinity norm
        grad = grad / (torch.norm(grad, float('inf')) + 1e-8)
        # Perform the SMART update.
        embeddings_perturbed = embeddings_perturbed + args.eta * grad
        # Project embeddings_perturbed back to the L_inf ball of radius epsilon centered at start_embeddings.
        embeddings_perturbed = start_embeddings + \
            torch.clamp(embeddings_perturbed - start_embeddings, -
                        args.epsilon, args.epsilon)

    # Calculating one more time for the final perturbatin loss, after we find
    # the most adversarial perturbation.
    logits = model(embeddings_perturbed, b_mask, is_embedding=True)

    return l_s(logits, orginal_logits, type=args.task_type)


def get_bregmman_loss(model_tilde: nn.Module, logits: torch.Tensor, b_ids: torch.Tensor, b_mask: torch.Tensor, args: Any):
    # TODO: unify the l_s calculation into a single function that also usable for regression task.
    # Disable grad as we never going to update logits_tilde.
    with torch.no_grad():
        logits_tilde = model_tilde(b_ids, b_mask)

    return l_s(logits, logits_tilde, type=args.task_type)


def update_model_tilde(model_tilde: nn.Module, model: nn.Module, beta: float):
    with torch.no_grad():
        for param_tilde, param_update in zip(model_tilde.parameters(), model.parameters()):
            param_tilde.mul_(beta)
            param_tilde.add_(param_update, alpha=1 - beta)


def standard_loss(output, b_labels, args):
    if args.task_type == "classifier":
        return F.cross_entropy(
            output, b_labels.view(-1), reduction='sum') / args.batch_size
    elif args.task_type == "regressor":
        return F.mse_loss(output.view(-1), b_labels.view(-1), reduction='sum') / args.batch_size


def pick_dataset(args, mode):
    data, num_labels = load_data(args, mode)

    if args.dataset == 'semeval' or args.dataset == 'quora':
        dataset = SentencePairDataset(data, args, is_test=(mode == 'test'))
    else:
        dataset = SentimentDataset(data, args, is_test=(mode == 'test'))

    return dataset, num_labels


def train(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.

    train_dataset, num_labels = pick_dataset(args, 'train')
    dev_dataset, _ = pick_dataset(args, 'valid')

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    if args.task_type == "classifier":
        model = BertSentimentClassifier(config)
    elif args.task_type == "regressor":
        model = BertRegressor(config)
    model = model.to(device)

    # Initialize the initial \tilde{\theta_1} when training with SMART (L1 in the algorithm).
    model_tilde = copy.deepcopy(model) if args.smart else None

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc, best_dev_mse = 0, float('inf')

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        train_iter = iter(train_dataloader)
        progress_bar = tqdm(total=len(train_dataloader),
                            desc=f'train-{epoch}', disable=TQDM_DISABLE)

        while True:
            try:
                # TODO: Encapsulate both vanilla AdamW and SMART into a function.
                if not args.smart:
                    batch = next(train_iter)
                    b_ids, b_mask, b_labels = (batch['token_ids'],
                                               batch['attention_mask'], batch['labels'])

                    b_ids = b_ids.to(device)
                    b_mask = b_mask.to(device)
                    b_labels = b_labels.to(device)
                    optimizer.zero_grad()
                    output = model(b_ids, b_mask)

                    loss = standard_loss(output, b_labels, args)

                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    num_batches += 1
                    progress_bar.update(1)
                else:
                    for _ in range(args.s):
                        # Sample a mini-batch B from X (L5)
                        batch = batch = next(train_iter)
                        b_ids, b_mask, b_labels = (batch['token_ids'],
                                                   batch['attention_mask'], batch['labels'])
                        b_ids = b_ids.to(device)
                        b_mask = b_mask.to(device)
                        b_labels = b_labels.to(device)

                        optimizer.zero_grad()

                        output = model(b_ids, b_mask)
                        base_loss = standard_loss(output, b_labels, args)

                        # Find the adversarial perturbation and add to loss (L6 - L10).
                        perturb_loss = get_perturb_loss(
                            model, b_ids, b_mask, output, args, device)

                        # Add trust region loss (equation (3)'s D_breg).
                        breg_loss = get_bregmman_loss(
                            model_tilde, output, b_ids, b_mask, args)

                        loss = base_loss + args.lambda_s * perturb_loss + args.mu * breg_loss

                        loss.backward()
                        optimizer.step()

                        train_loss += loss.item()
                        num_batches += 1
                        progress_bar.update(1)

                    # Update model tilde with momentum (L14).
                    update_model_tilde(model_tilde, model, args.beta)

            except StopIteration:
                break

        progress_bar.close()

        train_loss = train_loss / (num_batches)

        if args.task_type == "classifier":
            train_acc, train_f1, * \
                _ = model_eval(train_dataloader, model, device, args)
            dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device, args)

            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                save_model(model, optimizer, args, config, args.filepath)

            print(
                f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}, train f1 :: {train_f1 :.3f}, dev f1 :: {dev_f1 :.3f}")
        else:
            train_mse, train_r2, * \
                _ = model_eval(train_dataloader, model, device, args)
            dev_mse, dev_r2, *_ = model_eval(dev_dataloader, model, device, args)

            if dev_mse < best_dev_mse:
                best_dev_mse = dev_mse
                save_model(model, optimizer, args, config, args.filepath)

            print(
                f"Epoch {epoch}: train loss :: {train_loss :.3f}, train mse :: {train_mse :.3f}, dev mse :: {dev_mse :.3f}, train r2 :: {train_r2 :.3f}, dev r2 :: {dev_r2 :.3f}")


def test(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']
        if args.task_type == "classifier":
            model = BertSentimentClassifier(config)
        elif args.task_type == "regressor":
            model = BertRegressor(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"load model from {args.filepath}")

        dev_dataset, _ = pick_dataset(args, 'valid')
        dev_dataloader = DataLoader(
            dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

        test_dataset, _ = pick_dataset(args, 'test')
        test_dataloader = DataLoader(
            test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)

        dev_acc, dev_f1, dev_pred, dev_true, dev_sents, dev_sent_ids = model_eval(
            dev_dataloader, model, device, args)
        print('DONE DEV')
        test_pred, test_sents, test_sent_ids = model_test_eval(
            test_dataloader, model, device, args)
        print('DONE Test')
        with open(args.dev_out, "w+") as f:
            print(f"dev acc :: {dev_acc :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sent_ids, dev_pred):
                f.write(f"{p} , {s} \n")

        with open(args.test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sent_ids, test_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument(
        "--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-3)
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


def get_dataset_config(ds: str, args: Any):
    # TODO: Add 2 Quora and SemEval datasets.
    registry = {
        "sst": SimpleNamespace(
            dataset='sst',
            filepath='sst-classifier.pt',
            lr=args.lr,
            use_gpu=args.use_gpu,
            epochs=args.epochs,
            batch_size=args.batch_size,
            hidden_dropout_prob=args.hidden_dropout_prob,
            train='data/ids-sst-train.csv',
            dev='data/ids-sst-dev.csv',
            test='data/ids-sst-test-student.csv',
            option=args.option,
            dev_out='predictions/' + args.option + '-sst-dev-out.csv',
            test_out='predictions/' + args.option + '-sst-test-out.csv',
            smart=args.smart,
            lambda_s=args.lambda_s,
            sigma=args.sigma,
            epsilon=args.epsilon,
            eta=args.eta,
            tx=args.tx,
            s=args.s,
            mu=args.mu,
            beta=args.beta,
            task_type="classifier"
        ),
        "cfimdb": SimpleNamespace(
            dataset='cfimdb',
            filepath='cfimdb-classifier.pt',
            lr=args.lr,
            use_gpu=args.use_gpu,
            epochs=args.epochs,
            batch_size=8,
            hidden_dropout_prob=args.hidden_dropout_prob,
            train='data/ids-cfimdb-train.csv',
            dev='data/ids-cfimdb-dev.csv',
            test='data/ids-cfimdb-test-student.csv',
            option=args.option,
            dev_out='predictions/' + args.option + '-cfimdb-dev-out.csv',
            test_out='predictions/' + args.option + '-cfimdb-test-out.csv',
            smart=args.smart,
            lambda_s=args.lambda_s,
            sigma=args.sigma,
            epsilon=args.epsilon,
            eta=args.eta,
            tx=args.tx,
            s=args.s,
            mu=args.mu,
            beta=args.beta,
            task_type="classifier"
        ),
        "semeval": SimpleNamespace(
            dataset='semeval',
            filepath='semeval-regressor.pt',
            lr=args.lr,
            use_gpu=args.use_gpu,
            epochs=args.epochs,
            batch_size=8,
            hidden_dropout_prob=args.hidden_dropout_prob,
            train='data/sts-train.csv',
            dev='data/sts-dev.csv',
            test='data/sts-test-student.csv',
            option=args.option,
            dev_out='predictions/' + args.option + '-sts-dev-out.csv',
            test_out='predictions/' + args.option + '-sts-test-out.csv',
            smart=args.smart,
            lambda_s=args.lambda_s,
            sigma=args.sigma,
            epsilon=args.epsilon,
            eta=args.eta,
            tx=args.tx,
            s=args.s,
            mu=args.mu,
            beta=args.beta,
            task_type="regressor"
        ),
    }

    if ds not in registry:
        raise NotImplementedError

    return registry[ds]


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)

    for ds in args.dataset:
        print(f'Training Bert on {ds}...')

        config = get_dataset_config(ds, args)

        train(config)

        print(f'Evaluating on {ds}...')
        test(config)
