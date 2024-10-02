import torchattacks
import torch

def show_progress(title, index, maximum):
    """
    :param title:
    :param index:
    :param maximum:
    :return:
    """
    assert index >= 0 and maximum > 0
    # Show progress
    # (1) Get progress
    progress = min(int((index + 1) * 100.0 / maximum), 100)
    # (2) Show progress
    print('\r%s: [%s%s] %d%%' % (title, '>' * int(progress / 5), ' ' * (20 - int(progress / 5)), progress),
          end='\n' if progress == 100 else '')


def gaussian_kl_div(params1, params2='none', reduction='sum', average_batch=False):
    """
        0.5 * {
            sum_j [ log(var2)_j - log(var1)_j ]
            + sum_j [ (mu1 - mu2)^2_j / var2_j ]
            + sum_j (var1_j / var2_j)
            - K
        }
    :return:
    """
    assert reduction in ['sum', 'mean']
    # 1. Get params
    # (1) First
    mu1, std1 = params1
    # (2) Second
    if params2 == 'none':
        mu2 = torch.zeros(*mu1.size()).to(mu1.device)
        std2 = torch.ones(*std1.size()).to(std1.device)
    else:
        mu2, std2 = params2
    # 2. Calculate result
    result = 0.5 * (
        2 * (torch.log(std2) - torch.log(std1))
        + ((mu1 - mu2) / std2) ** 2
        + (std1 / std2) ** 2
        - 1)
    if reduction == 'sum':
        result = result.sum(dim=-1)
    else:
        result = result.mean(dim=-1)
    if average_batch:
        result = result.mean()
    # Return
    return result
