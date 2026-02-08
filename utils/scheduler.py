import math
import sys
import torch
import tqdm
import os
from .filters import get_factors
from .filters import MovingAvgTime


thismodule = sys.modules[__name__]

def get_schedule(noise_name, n_steps, check_pth, **kwargs):
    file_name = os.path.join(
        check_pth,
        f"{noise_name}_sched_{n_steps}.pt",
    )
    exist = os.path.exists(file_name)
    if exist:
        print("Already exist!")
    else:
        os.makedirs(check_pth, exist_ok=True)
        if noise_name in ["linear", "cosine", "zero"]:
            fn = getattr(thismodule, noise_name + "_schedule")
            noise_schedule = fn(n_steps)
            torch.save(noise_schedule, file_name)

        elif noise_name == "std":
            assert kwargs.get("train_dl") is not None
            fn = getattr(thismodule, noise_name + "_schedule")
            noise_schedule = fn(n_steps, kwargs.get("train_dl"))
            torch.save(noise_schedule, file_name)
        
        elif noise_name == "graph":
            assert kwargs.get("adj_matrix") is not None, "For 'graph' schedule, 'adj_matrix' must be provided in kwargs."
            fn = getattr(thismodule, noise_name + "_schedule")
            # 从kwargs中获取邻接矩阵和其他可选参数
            noise_schedule = fn(
                n_steps,
                adj_matrix=kwargs.get("adj_matrix"),
                noise_schedule_type=kwargs.get("noise_schedule_type", "cosine") # 默认使用cosine
            )
            torch.save(noise_schedule, file_name)
    return file_name

def linear_schedule(n_steps, min_beta=1e-4, max_beta=2e-2):
    betas = torch.linspace(min_beta, max_beta, n_steps)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    return {
        "alpha_bars": alpha_bars.float(),
        "beta_bars": None,
        "alphas": alphas.float(),
        "betas": betas.float(),
    }
    # return alpha_bars, beta_bars, alphas, betas

def cosine_schedule(n_steps: int, s: float = 0.008):
    """Cosine schedule for noise schedule.

    Args:
        n_steps (int): total number of steps.
        s (float, optional): tolerance. Defaults to 0.008.

    Returns:
        Dict[str, torch.Tensor]: noise schedule.
    """
    steps = n_steps + 1
    x = torch.linspace(0, n_steps, steps)
    alphas_cumprod = torch.cos(((x / n_steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0.001, 0.999)
    alphas = 1.0 - betas
    alphas_bars = torch.cumprod(alphas, dim=0)
    noise_schedule = dict(
        betas=betas,
        alphas=alphas,
        alpha_bars=alphas_bars,
        beta_bars=None,
    )
    return noise_schedule

def zero_schedule(n_steps):
    return {
        "alpha_bars": None,
        "beta_bars": None,
        "alphas": torch.ones(n_steps).float(),
        "betas": torch.zeros(n_steps).float(),
    }

def std_schedule(n_steps, train_dl):
    batch = next(iter(train_dl))
    seq_length = batch["x"].shape[1]
    all_x = torch.concat([batch["x"] for batch in train_dl])

    orig_std = torch.sqrt(torch.var(all_x, dim=1, unbiased=False) + 1e-5)
    kernel_list = get_factors(seq_length)
    step_ratios, all_K = [], []
    all_K = torch.stack(
        [MovingAvgTime(j, seq_length=seq_length, stride=j).K for j in kernel_list]
    )
    all_K = (
        all_K.flatten(1).unsqueeze(0).permute(0, 2, 1)
    )  # [1, seq_len*seq_len, n_factors]
    interp_all_K = (
        torch.nn.functional.interpolate(
            all_K, size=n_steps + 1, mode="linear", align_corners=True
        )
        .squeeze()
        .reshape(seq_length, seq_length, -1)
        .permute(2, 0, 1)
    )  # [n_steps + 1, seq_len, seq_len]

    # first step is original data
    interp_all_K = interp_all_K[1:]

    for j in tqdm.tqdm(range(len(interp_all_K))):
        x_avg = interp_all_K[j] @ all_x
        std_avg = torch.sqrt(torch.var(x_avg, dim=1, unbiased=False) + 1e-5)
        ratio = std_avg / orig_std
        step_ratios.append(ratio.mean(dim=0))
    all_ratio = torch.stack(step_ratios)  # [n_factors, seq_chnl]

    print(interp_all_K.shape)
    print(torch.sqrt(1 - all_ratio**2).float().shape)
    return {
        "alpha_bars": None,
        "beta_bars": None,
        "alphas": interp_all_K,
        "betas": torch.sqrt(1 - all_ratio**2).float(),
    }

# ==================== 新增的 graph_schedule 函数 ====================
def graph_schedule(n_steps, adj_matrix, noise_schedule_type="cosine"):
    """
    生成一个基于图卷积的噪声计划，其中每个元素是 A^t。
    """
    print(f"Generating graph schedule with pre-computed matrix powers and '{noise_schedule_type}' noise levels.")
    
    # 1. 从标准的标量计划中获取 betas 和 alpha_bars
    if noise_schedule_type == "cosine":
        scalar_schedule = cosine_schedule(n_steps)
    else: # linear
        scalar_schedule = linear_schedule(n_steps)
    betas = scalar_schedule["betas"]
    alpha_bars = scalar_schedule["alpha_bars"]

    # 2. 预计算矩阵的幂: A^1, A^2, ..., A^n_steps
    print("Pre-computing matrix powers A^1 to A^N...")
    graph_conv_powers = []
    current_power = adj_matrix.clone() # A^1
    graph_conv_powers.append(current_power)
    
    # 使用 tqdm 显示进度条
    for _ in tqdm.tqdm(range(1, n_steps), desc="Computing A^t"):
        current_power = current_power @ adj_matrix # A^(i+1) = A^i @ A
        graph_conv_powers.append(current_power)
    
    # 将列表堆叠成一个张量
    graph_conv_matrices = torch.stack(graph_conv_powers, dim=0)

    # 3. 组装并返回最终的噪声计划
    return {
        "alpha_bars": alpha_bars,
        "beta_bars": None,
        "alphas": graph_conv_matrices.float(),
        "betas": betas.float(),
    }