from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        return F.mse_loss(p.view(-1), q.view(-1), reduction='mean')


def get_perturb_loss(task: nn.Module, b_ids: torch.Tensor, b_mask: torch.Tensor, orginal_logits: torch.Tensor, args: Any, ls_type: str):
    # Use the same device as the input tensor.
    device = b_ids.device

    # In addition to the standard cross-entropy loss, we also add the SMART loss.
    # Compute the embedding of the batch
    start_embeddings = task.model.embed(b_ids)

    # Perturb the embedding with Gaussian noise.
    embeddings_perturbed: torch.Tensor = start_embeddings + \
        torch.normal(0, args.sigma, start_embeddings.size()).to(device)

    # Loop until tx iterations have been performed
    for _ in range(args.tx):
        # Compute the gradient of the loss with respect to the perturbed embedding.
        embeddings_perturbed.requires_grad_()
        logits = task.forward_with_embedding(embeddings_perturbed, b_mask)

        # Use symmetrizied KL divergence as the loss function.
        # TODO: unify the l_s calculation into a single function that also usable for regression task.
        loss_perturbed = l_s(logits, orginal_logits, type=ls_type)

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
    logits = task.forward_with_embedding(embeddings_perturbed, b_mask)

    return l_s(logits, orginal_logits, type=ls_type)


def get_perturb_loss_for_pair(task: nn.Module, b_ids1: torch.Tensor, b_mask1: torch.Tensor, b_ids2: torch.Tensor, b_mask2: torch.Tensor, orginal_logits: torch.Tensor, args: Any, ls_type: str):
    # Use the same device as the input tensor.
    device = b_ids1.device

    # We can perturb both embeddings at the same time.
    # Compute the embedding of the batch
    start_embeddings_1 = task.model.embed(b_ids1)
    start_embeddings_2 = task.model.embed(b_ids2)

    # Perturb the embedding with Gaussian noise.
    embeddings_perturbed_1: torch.Tensor = start_embeddings_1 + \
        torch.normal(0, args.sigma, start_embeddings_1.size()).to(device)
    embeddings_perturbed_2: torch.Tensor = start_embeddings_2 + \
        torch.normal(0, args.sigma, start_embeddings_2.size()).to(device)

    # Loop until tx iterations have been performed
    for _ in range(args.tx):
        # Compute the gradient of the loss with respect to the perturbed embedding.
        embeddings_perturbed_1.requires_grad_()
        embeddings_perturbed_2.requires_grad_()

        logits = task.forward_with_embedding(
            embeddings_perturbed_1, b_mask1,
            embeddings_perturbed_2, b_mask2)

        # Use symmetrizied KL divergence as the loss function.
        loss_perturbed = l_s(logits, orginal_logits, type=ls_type)

        grad_1 = torch.autograd.grad(
            loss_perturbed, embeddings_perturbed_1, retain_graph=True)[0]
        grad_2 = torch.autograd.grad(
            loss_perturbed, embeddings_perturbed_2)[0]

        # Normalize the gradient by infinity norm
        grad_1 = grad_1 / (torch.norm(grad_1, float('inf')) + 1e-8)
        grad_2 = grad_2 / (torch.norm(grad_2, float('inf')) + 1e-8)
        # Perform the SMART update.
        embeddings_perturbed_1 = embeddings_perturbed_1 + args.eta * grad_1
        embeddings_perturbed_2 = embeddings_perturbed_2 + args.eta * grad_2
        # Project embeddings_perturbed back to the L_inf ball of radius epsilon centered at start_embeddings.
        embeddings_perturbed_1 = start_embeddings_1 + \
            torch.clamp(embeddings_perturbed_1 - start_embeddings_1, -
                        args.epsilon, args.epsilon)
        embeddings_perturbed_2 = start_embeddings_2 + \
            torch.clamp(embeddings_perturbed_2 - start_embeddings_2, -
                        args.epsilon, args.epsilon)

    # Calculating one more time for the final perturbatin loss, after we find
    # the most adversarial perturbation.
    logits = task.forward_with_embedding(
        embeddings_perturbed_1, b_mask1, embeddings_perturbed_2, b_mask2)

    return l_s(logits, orginal_logits, type=ls_type)


def get_bregmman_loss(task_tilde: nn.Module, batch: torch.Tensor, logits: torch.Tensor, ls_type: str):
    # Disable grad as we never going to update logits_tilde.
    with torch.no_grad():
        logits_tilde = task_tilde(batch)

    return l_s(logits, logits_tilde, type=ls_type)


def update_model_tilde(model_tildes: List[nn.Module], models: List[nn.Module], beta: float, fraction: float):
    # Use a different beta as training progresses. Scale down when training move beyond the first 10%.
    beta = beta if fraction < 0.1 else 0.1 * beta

    # Share parameters (specifically the base BERT) should be updated only once for all tasks.
    updated_params = set()

    with torch.no_grad():
        for model_tilde, model in zip(model_tildes, models):
            for param_tilde, param_update in zip(model_tilde.parameters(), model.parameters()):
                if id(param_update) in updated_params:
                    continue

                param_tilde.mul_(beta).add_(param_update, alpha=1 - beta)
                updated_params.add(id(param_update))
