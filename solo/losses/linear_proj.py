import torch

def l2_norm(x):
    input_size = x.size()
    buffer = torch.pow(x, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(x, norm.view(-1, 1).expand_as(x))
    output = _output.view(input_size)
    return output

def linear_projection_loss_func(
    z1: torch.Tensor, z2: torch.Tensor, nb_classes: int = 256
) -> torch.Tensor:

    # initialize linear projection model
    linear = torch.nn.Linear(in_features=z1.shape[-1],
                             out_features=z1.shape[0], bias=False)
    # inference_sample -> z2
    labels_bin = torch.diag_embed(torch.ones(z1.shape[0],
                                             z1.shape[0]))[0].to(z1.device)
    labels = labels_bin.argmax(dim=0)

    # solve with exact solution and append new weights
    linear.weight = torch.nn.Parameter(torch.matmul(torch.pinverse(l2_norm(z1)), labels_bin).T.detach())
    # w = torch.matmul(torch.pinverse(l2_norm(z1)), labels_bin).T

    # run inference and calculate cross-entropy loss
    z2 = l2_norm(z2)
    out = torch.nn.functional.softmax(linear(z2), dim=-1)
    # out = torch.nn.functional.softmax(torch.matmul(z2, w.T), dim=-1).detach().half()
    loss = torch.nn.functional.cross_entropy(input=out, target=labels)
    return loss