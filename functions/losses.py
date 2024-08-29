import torch


def noise_estimation_loss(model,
                          x0: torch.Tensor,
                          x_sr: torch.Tensor,
                          t: torch.LongTensor,
                          e: torch.Tensor,
                          b: torch.Tensor, keepdim=False):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    y_t = x0 * a.sqrt() + e * (1.0 - a).sqrt()

    x_input = torch.cat([y_t, x_sr], dim=1)
    output = model(x_input, t.float())
    if keepdim:
        return (e - output).square().sum(dim=(1, 2, 3))
    else:
        return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


loss_registry = {
    'simple': noise_estimation_loss,
}
