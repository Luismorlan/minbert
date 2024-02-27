from typing import Any

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
        return F.mse_loss(p, q, reduction='mean')


def get_perturb_loss(model: nn.Module, b_ids: torch.Tensor, b_mask: torch.Tensor, orginal_logits: torch.Tensor, args: Any, device: Any, predict_fn: str = ''):
    # In addition to the standard cross-entropy loss, we also add the SMART loss.
    # Compute the embedding of the batch
    start_embeddings = model.embed(b_ids)
    # Different task require different predict function
    predict_fn = getattr(model, predict_fn, model.__call__)

    # Perturb the embedding with Gaussian noise.
    embeddings_perturbed: torch.Tensor = start_embeddings + \
        torch.normal(0, args.sigma, start_embeddings.size()).to(device)

    # Loop until tx iterations have been performed
    for _ in range(args.tx):
        # Compute the gradient of the loss with respect to the perturbed embedding.
        embeddings_perturbed.requires_grad_()
        logits = predict_fn(embeddings_perturbed, b_mask, is_embedding=True)

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
    logits = predict_fn(embeddings_perturbed, b_mask, is_embedding=True)

    return l_s(logits, orginal_logits, type=args.task_type)


def get_perturb_loss_for_pair(model: nn.Module, b_ids1: torch.Tensor, b_mask1: torch.Tensor, b_ids2: torch.Tensor, b_mask2: torch.Tensor, orginal_logits: torch.Tensor, args: Any, device: Any, predict_fn: str = ''):
    # We can perturb both embeddings at the same time.
    # Compute the embedding of the batch
    start_embeddings_1 = model.embed(b_ids1)
    start_embeddings_2 = model.embed(b_ids2)
    # Different task require different predict function
    predict_fn = getattr(model, predict_fn, model.__call__)

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
        logits = predict_fn(
            embeddings_perturbed_1, b_mask1,
            embeddings_perturbed_2, b_mask2,
            is_embedding=True)

        # Use symmetrizied KL divergence as the loss function.
        # TODO: unify the l_s calculation into a single function that also usable for regression task.
        loss_perturbed = l_s(logits, orginal_logits, type=args.task_type)

        grad_1 = torch.autograd.grad(
            loss_perturbed, embeddings_perturbed_1)[0]
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
    logits = predict_fn(
        embeddings_perturbed_1, b_mask1,
        embeddings_perturbed_2, b_mask2,
        is_embedding=True)

    return l_s(logits, orginal_logits, type=args.task_type)


def get_bregmman_loss(model_tilde: nn.Module, logits: torch.Tensor, b_ids: torch.Tensor, b_mask: torch.Tensor, args: Any, predict_fn: str = ''):
    predict_fn = getattr(model_tilde, predict_fn, model_tilde.__call__)
    # TODO: unify the l_s calculation into a single function that also usable for regression task.
    # Disable grad as we never going to update logits_tilde.
    with torch.no_grad():
        logits_tilde = predict_fn(b_ids, b_mask)

    return l_s(logits, logits_tilde, type=args.task_type)


def get_bregmman_loss_for_pair(model_tilde: nn.Module, logits: torch.Tensor, b_ids1: torch.Tensor, b_mask1: torch.Tensor, b_ids2: torch.Tensor, b_mask2: torch.Tensor, args: Any, predict_fn: str = ''):
    predict_fn = getattr(model_tilde, predict_fn, model_tilde.__call__)
    with torch.no_grad():
        logits_tilde = predict_fn(b_ids1, b_mask1, b_ids2, b_mask2)

    return l_s(logits, logits_tilde, type=args.task_type)


def update_model_tilde(model_tilde: nn.Module, model: nn.Module, beta: float):
    with torch.no_grad():
        for param_tilde, param_update in zip(model_tilde.parameters(), model.parameters()):
            param_tilde.mul_(beta)
            param_tilde.add_(param_update, alpha=1 - beta)
