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
from typing import Any, List
from itertools import cycle
import copy
import utils

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from task import TQDM_DISABLE, Task, SentimentClassificationTask, ParaphraseDetectionTask, SemanticTextualSimilarityTask
from loss import update_model_tilde

from tqdm import tqdm
from dataclasses import dataclass

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from evaluation import model_eval_multitask, model_eval_test_multitask
from torch.profiler import profile, record_function, ProfilerActivity


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


@dataclass
class TaskInfo:
    name: str
    model_path: str
    test_pred_path: str
    train_dataloader: DataLoader
    dev_dataloader: DataLoader
    test_dataloader: DataLoader


TASK_REGISTRY = {}

# Get all parameters from a list of model, and make sure they are unique
# (shared params will be counted only once).


def get_unique_params(models: List[nn.Module]):
    unique_ids = set()
    res = []
    for model in models:
        for param in model.parameters():
            if id(param) in unique_ids:
                continue

            # Add to parameter set
            res.append(param)
            unique_ids.add(id(param))

    return res


def init_task_tilde(tasks: List[nn.Module]) -> List[nn.Module]:
    '''Initialize the model tilde for SMART update. 

    It's important to keep the same Bert Base model parameter sharing after copy operation.
    '''
    copies = [copy.deepcopy(task) for task in tasks]
    i = 1
    while i < len(copies):
        copies[i].model = copies[0].model
        i += 1
    return copies


class Trainer:
    def __init__(self, args: Any, tasks: List[Task]) -> None:
        self.args = args

        # Each task contains the base BERT model.
        self.tasks = tasks

    def _loop(self, train_iter, device, optimizer, train_loss, num_batches, progress_bar, tasks_tilde, epoch, prof=None):
        steps = 0
        while True:
            try:
                steps += 1
                if prof is not None:
                    prof.step()
                if not self.args.smart:
                    batches = next(train_iter)
                    loss = 0
                    optimizer.zero_grad()
                    for batch, task in zip(batches, self.tasks):
                        batch = utils.move_batch(batch, device)

                        pred = task.forward(batch)
                        loss += task.loss(batch, pred)

                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    num_batches += 1
                    progress_bar.update(1)
                else:
                    for _ in range(self.args.s):
                        loss = 0
                        optimizer.zero_grad()
                        batches = next(train_iter)
                        for batch, task, task_tilde in zip(batches, self.tasks, tasks_tilde):
                            batch = utils.move_batch(batch, device)

                            # 1. task specific loss.
                            pred = task.forward(batch)
                            loss += task.loss(batch, pred)

                            # 2. perturbed loss to be robust to noise.
                            loss += self.args.lambda_s * \
                                task.perturbed_loss(batch, pred)

                            # 3. bregmman loss to not deviate too much from original model.
                            loss += self.args.mu * \
                                task.bregmman_loss(batch, pred, task_tilde)

                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()
                        num_batches += 1
                        progress_bar.update(1)

                    update_model_tilde(
                        tasks_tilde, self.tasks, self.args.beta, epoch / self.args.epochs)

                # hardcode 5 as profiling steps
                if prof is not None and steps > 5:
                    return num_batches
            except StopIteration:
                return num_batches

    def train(self, device: str):
        for task in self.tasks:
            task.to(device)
            task.train()

        best_dev_metrics = [0,] * len(self.tasks)  # higher the better
        acc_dev = [0,] * len(self.tasks)  # higher the better

        # ith element's j th epoch.
        all_train_metrics = [[] for _ in self.tasks]
        all_dev_metrics = [[] for _ in self.tasks]

        optimizer = AdamW(get_unique_params(self.tasks), lr=self.args.lr)

        train_dataloaders = [task.train_dataloader for task in self.tasks]
        dev_dataloaders = [task.dev_dataloader for task in self.tasks]

        max_len, longest_dl = max(
            [(len(dl), dl) for dl in train_dataloaders], key=lambda x: x[0])

        # Align on the longest data loader, cycle other short data loaders. This is essentially
        # to upsample small dataset so that it doesn't overfit that much.
        #
        # TODO: Add additional strategy for just simply concate dataset instead of cycle.
        aligned_train_dataloaders = [
            cycle(dl) if dl != longest_dl else dl for dl in train_dataloaders]

        tasks_tilde = init_task_tilde(self.tasks)

        for epoch in range(self.args.epochs):
            num_batches = 0
            train_loss = 0

            progress_bar = tqdm(
                desc=f'train-{epoch}', disable=TQDM_DISABLE, total=max_len)
            # we zip the dataloaders together to train on all tasks at the same time
            train_iter = iter(zip(*aligned_train_dataloaders))

            if self.args.profile:
                with torch.profiler.profile(
                        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        # Comment out the following line to generate trace.json, which can be viewed in chrome://tracing
                        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/minbert'),
                        record_shapes=True,
                        profile_memory=True,
                        with_stack=True
                ) as prof:
                        num_batches = self._loop(train_iter, device, optimizer, train_loss,
                       num_batches, progress_bar, tasks_tilde, epoch, prof)
                        
                print(">"*20 + "CPU Profile" + "<"*20)
                print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
                print(">"*20 + "CUDA Profile" + "<"*20)
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
                # Uncomment the following line to export the trace for chrome/edge
                # prof.export_chrome_trace("trace.json")

                return

            else:
                num_batches = self._loop(train_iter, device, optimizer, train_loss,
                        num_batches, progress_bar, tasks_tilde, epoch)

            progress_bar.close()

            train_loss = train_loss / (num_batches)

            print(f"\n>>>Epoch {epoch}:\n train loss :: {train_loss :.3f}")

            for i, (task, train_dataloader, dev_dataloader) in enumerate(zip(self.tasks, train_dataloaders, dev_dataloaders)):
                print(f"\nEvaluating {task.name} task on training set")
                train_metric = task.evaluate(train_dataloader).metric
                print(f"\nEvaluating {task.name} task on dev set")
                dev_metric = task.evaluate(dev_dataloader).metric
                acc_dev[i] = dev_metric

                # Store the metrics for later analysis.
                all_train_metrics[i].append(train_metric)
                all_dev_metrics[i].append(dev_metric)

            # TODO: how to weight these dev metrics across different tasks?
            # TODO: re-enable the best model parameters.
            if np.average(acc_dev) > np.average(best_dev_metrics):
                best_dev_metrics = acc_dev
                for task in self.tasks:
                    torch.save(task, TASK_REGISTRY[task.name].model_path)

        # Write the final metrics to a file.
        with open('final_metrics.txt', 'a') as f:
            f.write("\n===============\n")
            all_tasks = "_".join([task.name for task in self.tasks])
            smart = "" if not args.smart else "-smart"
            for i, task in enumerate(self.tasks):
                f.write(
                    f"\n{task.name}-{args.option}-{args.epochs}-{args.lr}-{all_tasks}{smart}\n")
                f.write(f"\ntrain metric: {all_train_metrics[i]}\n")
                f.write(f"\ndev metric: {all_dev_metrics[i]}\n")
            f.write("\n===============\n")


def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    bert = BertModel.from_pretrained('bert-base-uncased').to(device)

    # Pretrain mode does not require updating BERT paramters.
    for param in bert.parameters():
        if args.option == 'pretrain':
            param.requires_grad = False
        elif args.option == 'finetune':
            param.requires_grad = True

    tasks = []
    for task_name in args.tasks:
        task_info: TaskInfo = TASK_REGISTRY[task_name]

        if task_info.name == "sst":
            tasks.append(SentimentClassificationTask(
                hidden_size=BERT_HIDDEN_SIZE,
                num_labels=5,
                model=bert,
                name="sst",
                train_dataloader=task_info.train_dataloader,
                dev_dataloader=task_info.dev_dataloader,
                args=args))
        if task_info.name == "quora":
            tasks.append(ParaphraseDetectionTask(
                hidden_size=BERT_HIDDEN_SIZE,
                model=bert,
                name="quora",
                train_dataloader=task_info.train_dataloader,
                dev_dataloader=task_info.dev_dataloader,
                args=args))
        if task_info.name == "sts":
            tasks.append(SemanticTextualSimilarityTask(
                hidden_size=BERT_HIDDEN_SIZE,
                model=bert,
                name="sts",
                train_dataloader=task_info.train_dataloader,
                dev_dataloader=task_info.dev_dataloader,
                args=args))
        if task_info.name == "cfimdb":
            tasks.append(SentimentClassificationTask(
                hidden_size=BERT_HIDDEN_SIZE,
                num_labels=2,
                model=bert,
                name="cfimdb",
                train_dataloader=task_info.train_dataloader,
                dev_dataloader=task_info.dev_dataloader,
                args=args))

    assert len(tasks) > 0, "No task is loaded."

    trainer = Trainer(args, tasks)
    trainer.train(device)


def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        for task_name in args.tasks:
            task: Task = torch.load(
                TASK_REGISTRY[task_name].model_path).to(device)

            # TODO: Change based on task to evaluate
            res = task.evaluate(
                TASK_REGISTRY[task_name].test_dataloader, is_hidden=True)
            with open(TASK_REGISTRY[task_name].test_pred_path, "w+") as f:
                print(f"Writing prediction output for {task_name}")
                f.write(f"id \t Predicted_Sentiment \n")
                for p, s in zip(res.ids, res.pred):
                    f.write(f"{p} , {s} \n")
                print(f"Done writing for {task_name}")


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

    parser.add_argument("--cfimdb_train", type=str,
                        default="data/ids-cfimdb-train.csv")
    parser.add_argument("--cfimdb_dev", type=str,
                        default="data/ids-cfimdb-dev.csv")
    parser.add_argument("--cfimdb_test", type=str,
                        default="data/ids-cfimdb-test-student.csv")

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

    parser.add_argument("--cfimdb_dev_out", type=str,
                        default="predictions/sts-dev-output.csv")
    parser.add_argument("--cfimdb_test_out", type=str,
                        default="predictions/sts-test-output.csv")

    parser.add_argument(
        "--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    parser.add_argument("--smart", action='store_true',
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
    # cfimdb is available in classifier.py
    parser.add_argument("--tasks", nargs="*", choices=[
                        'sst', 'quora', 'sts', 'cfimdb'], default=["sst"], help="List of datasets that can be used to train or finetune.")
    parser.add_argument("--mu", type=float,
                        help="Coefficient for Bregmma loss", default=1)
    # TODO: use a rate schedule for beta. In the paper it is decreasing to 0.999 after 10% of training.
    parser.add_argument(
        "--beta", type=float, help="Coefficient for momentum of theta tilde", default=0.99)
    # TODO: actually implement this
    parser.add_argument("--contrastive", action='store_true',
                        help="Use contrastive loss fromm https://arxiv.org/pdf/2104.08821.pdf")
    parser.add_argument(
        "--tau", type=float, help="Tau coefficient for the contrastive loss", default=0.1)
    parser.add_argument(
        "--profile", action='store_true', help="Profile the training process")

    args = parser.parse_args()
    return args


def register_tasks(args: Any):
    """Load all datasets and register them by corresponding task name."""
    sst_train_data, para_train_data, sts_train_data, cfimdb_train_data = load_multitask_data(
        args.sst_train, args.para_train, args.sts_train, args.cfimdb_train, split='train')
    sst_test_data, para_test_data, sts_test_data, cfimdb_test_data = \
        load_multitask_data(args.sst_test, args.para_test,
                            args.sts_test, args.cfimdb_test, split='test')
    sst_dev_data, para_dev_data, sts_dev_data, cfimdb_dev_data = load_multitask_data(
        args.sst_dev, args.para_dev, args.sts_dev, args.cfimdb_dev, split='dev')

    # SST dataset.
    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
    sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)

    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)
    para_test_data = SentencePairTestDataset(para_test_data, args)

    sts_train_data = SentencePairDataset(
        sts_train_data, args, isRegression=True)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)
    sts_test_data = SentencePairTestDataset(sts_test_data, args)

    cfimdb_train_data = SentenceClassificationDataset(cfimdb_train_data, args)
    cfimdb_dev_data = SentenceClassificationDataset(cfimdb_dev_data, args)
    cfimdb_test_data = SentenceClassificationTestDataset(
        cfimdb_test_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn, num_workers=2)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn, num_workers=2)
    sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=sst_test_data.collate_fn)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                       collate_fn=para_train_data.collate_fn, num_workers=2)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=para_dev_data.collate_fn, num_workers=2)
    para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_test_data.collate_fn)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn, num_workers=2)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn, num_workers=2)
    sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                     collate_fn=sts_test_data.collate_fn)

    cfimdb_train_dataloader = DataLoader(cfimdb_train_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=cfimdb_train_data.collate_fn, num_workers=2)
    cfimdb_dev_dataloader = DataLoader(cfimdb_dev_data, shuffle=False, batch_size=args.batch_size,
                                       collate_fn=cfimdb_dev_data.collate_fn, num_workers=2)
    cfimdb_test_dataloader = DataLoader(cfimdb_test_data, shuffle=True, batch_size=args.batch_size,
                                        collate_fn=cfimdb_test_data.collate_fn)

    TASK_REGISTRY["sst"] = TaskInfo(
        name="sst",
        model_path=f'sst-{args.option}-{args.epochs}-{args.lr}-multitask.pt',
        test_pred_path=f'predictions/sst-{args.option}-{args.epochs}-{args.lr}-multitask-test-pred.csv',
        train_dataloader=sst_train_dataloader,
        dev_dataloader=sst_dev_dataloader,
        test_dataloader=sst_test_dataloader
    )
    TASK_REGISTRY["quora"] = TaskInfo(
        name="quora",
        model_path=f'quora-{args.option}-{args.epochs}-{args.lr}-multitask.pt',
        test_pred_path=f'predictions/quora-{args.option}-{args.epochs}-{args.lr}-multitask-test-pred.csv',
        train_dataloader=para_train_dataloader,
        dev_dataloader=para_dev_dataloader,
        test_dataloader=para_test_dataloader
    )
    TASK_REGISTRY["sts"] = TaskInfo(
        name="sts",
        model_path=f'sts-{args.option}-{args.epochs}-{args.lr}-multitask.pt',
        test_pred_path=f'predictions/sts-{args.option}-{args.epochs}-{args.lr}-multitask-test-pred.csv',
        train_dataloader=sts_train_dataloader,
        dev_dataloader=sts_dev_dataloader,
        test_dataloader=sts_test_dataloader
    )
    TASK_REGISTRY["cfimdb"] = TaskInfo(
        name="cfimdb",
        model_path=f'cfimdb-{args.option}-{args.epochs}-{args.lr}-multitask.pt',
        test_pred_path=f'predictions/cfimdb-{args.option}-{args.epochs}-{args.lr}-multitask-test-pred.csv',
        train_dataloader=cfimdb_train_dataloader,
        dev_dataloader=cfimdb_dev_dataloader,
        test_dataloader=cfimdb_test_dataloader
    )


if __name__ == "__main__":
    args = get_args()
    # Save path.
    args.filepath = f'{args.option}-{args.epochs}-{args.lr}-multitask.pt'
    seed_everything(args.seed)  # Fix the seed for reproducibility.

    register_tasks(args)

    train_multitask(args)
    test_multitask(args)
