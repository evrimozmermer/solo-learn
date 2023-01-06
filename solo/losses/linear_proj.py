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

    linear = torch.nn.Linear(in_features=z1.shape[-1],
                             out_features=z1.shape[0], bias=False)
    # inference_sample -> z2
    labels_bin = torch.diag_embed(torch.ones(z1.shape[0],
                                             z1.shape[0]))[0].to(z1.device)
    labels = labels_bin.argmax(dim=0)

    # solve with exact solution and append new weights
    with torch.no_grad():
        linear.weight = torch.nn.Parameter(torch.matmul(torch.pinverse(l2_norm(z1)), labels_bin).T.detach())

    # run inference and calculate cross-entropy loss

    loss = torch.nn.functional.cross_entropy(input=linear(l2_norm(z2)), target=labels)
    return loss