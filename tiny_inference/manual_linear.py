from __future__ import annotations

import torch
import torch.nn.functional as F


def torch_causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> torch.Tensor:
    """
    Decode 阶段单步卷积：每次只处理 1 个新 token，用滑动窗口更新 conv_state，
    替代 prefill 时对整段序列的完整 conv1d。

    参数介绍
    ----
    hidden_states : (batch, hidden_size, 1)
        新 token 经过线性投影后的表示（Q/K/V 拼接）
    conv_state    : (batch, hidden_size, kernel_size)
        因果卷积的滑动窗口快照，被原地更新（.copy_）
    weight        : 卷积核权重，形状 (hidden_size, kernel_size)
    bias          : 可选偏置

    实现提示
    --------
    可用函数：
      - torch.cat([a, b], dim=-1)
          在时间轴（最后一维）拼接两个张量；
          注意拼接前先 .to(weight.dtype) 统一精度
      - conv_state.copy_(x)
          原地将 conv_state 更新为 x（[:, :, -state_len:] 取末尾 kernel_size 帧）
      - F.conv1d(input, weight, bias, padding=0, groups=hidden_size)
          逐通道卷积（groups=hidden_size 表示每个通道独立卷积）；
          weight 形状需为 (hidden_size, 1, kernel_size)，即需要 weight.unsqueeze(1)
      - F.silu(x)
          Sigmoid Linear Unit 激活函数

    步骤：
    1. 将 conv_state（旧窗口）与 hidden_states（新 token）在 dim=-1 拼接 → shape (batch, hidden_size, kernel_size+1)
    2. 用 .copy_ 原地更新 conv_state 为拼接结果末尾的 kernel_size 帧
    3. 对拼接结果做 F.conv1d，取输出末尾 seq_len 帧，经 F.silu 激活
    4. 转回原始 dtype 返回
    """
    # ===== TODO: KV Cache - Linear Attention 单步卷积更新 (START) =====
    # 1. 将 conv_state（旧窗口）与 hidden_states（新 token）在 dim=-1 拼接
    # 确保精度一致
    hidden_states = hidden_states.to(weight.dtype)
    # 拼接后的形状：(batch, hidden_size, kernel_size+1)
    combined = torch.cat([conv_state, hidden_states], dim=-1)
    
    # 2. 用 .copy_ 原地更新 conv_state 为拼接结果末尾的 kernel_size 帧
    kernel_size = conv_state.shape[-1]
    conv_state.copy_(combined[..., -kernel_size:])
    
    # 3. 对拼接结果做 F.conv1d，取输出末尾 seq_len 帧，经 F.silu 激活
    # 逐通道卷积：weight 形状应为 (hidden_size, 1, kernel_size)
    # 输入 combined 形状为 (batch, hidden_size, kernel_size+1)
    # groups=hidden_size 表示每个通道独立卷积
    weight_reshaped = weight.unsqueeze(1)
    # 执行逐通道卷积，groups 应等于输入通道数
    conv_output = F.conv1d(combined, weight_reshaped, bias, padding=0, groups=hidden_states.shape[1])
    # 取末尾 1 个时间步的输出并激活
    conv_output = F.silu(conv_output[..., -1:])
    
    # 4. 转回原始 dtype 返回
    return conv_output.to(hidden_states.dtype)
    # ===== TODO: KV Cache - Linear Attention 单步卷积更新 (END) =====


def torch_recurrent_gated_delta_rule(
    query, key, value, g, beta,
    initial_state, output_final_state,
    use_qk_l2norm_in_kernel=False,
):
    """
    Decode 阶段逐 token 递推，从 initial_state（= 上一步的 recurrent_state）出发，
    按 Gated Delta Rule 更新记忆矩阵 S，输出本步注意力结果。

    与 prefill 时使用的 torch_chunk_gated_delta_rule 在数学上完全等价，
    但改为纯 Python 循环逐步递推，适合 seq_len=1 的 decode 场景。

    参数介绍
    ----
    query, key, value : (batch, seq_len, num_heads, head_dim)，此处 seq_len=1
    g                 : 遗忘门参数，经 .exp() 后得到衰减因子 e^α（< 1，让旧状态衰减）
    beta              : 更新门参数，经 .sigmoid() 后控制写入强度
    initial_state     : 上一步保存的 recurrent_state，形状 (batch, num_heads, k_dim, v_dim)
                        prefill 时由 torch_chunk_gated_delta_rule 输出；decode 时逐步传递
    output_final_state: 是否返回更新后的状态（use_cache=True 时为 True，更新后写回缓存）
    """
    initial_dtype = query.dtype
    # 若需要，对 Q/K 做 L2 归一化（稳定数值）
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    # 转为 (batch, num_heads, seq_len, head_dim) 布局，统一用 float32 计算
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale  # 缩放 query，防止点积过大

    # 输出缓冲区；记忆矩阵 S 从 initial_state 初始化（若无则全零）
    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim).to(value)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]   # (batch, heads, k_dim)
        k_t = key[:, :, i]
        v_t = value[:, :, i]   # (batch, heads, v_dim)
        # 遗忘旧状态：S = exp(g_t) * S，g_t 广播到 (batch, heads, k_dim, v_dim)
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)  # 写入强度，(batch, heads, 1)

        last_recurrent_state = last_recurrent_state * g_t
        # 用 k_t 从记忆矩阵中检索历史 value：kv_mem = (S * k_t).sum(k_dim)
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        # 写入误差：目标 v 与检索值之差，乘以写入强度 beta
        delta = (v_t - kv_mem) * beta_t
        # 外积更新记忆矩阵：S += k_t ⊗ delta，形状 (batch, heads, k_dim, v_dim)
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        # 用 q_t 从更新后的记忆矩阵中读出本步输出
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
    # 转回 (batch, seq_len, num_heads, v_dim) 并恢复原始精度
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def apply_mask_to_padding_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states


def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def qwen3_5_linear_attn_forward(
    linear_attn_module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cache_params=None, # Qwen3_5DynamicCache 对象，存放 conv_state 和 recurrent_state；利用它实现缓存读写
    cache_position=None, # 当前 tokens 的绝对位置，用于判断是否进入 decode 路径
    layer_idx: int = 0, # 当前层索引，用于定位缓存槽位
) -> torch.Tensor:
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape

    # ===== TODO: KV Cache - Linear Attention 判断是否走 decode 单步路径 (START) =====
    # 四个条件同时满足时走 decode 单步路径（即 use_precomputed_states=True），否则走 prefill 完整路径
    # 满足条件：
    # 1. 存在缓存对象
    # 2. 序列长度为1（decode阶段每次只处理1个token）
    # 3. 存在缓存位置信息
    # 4. 缓存中已有之前的状态
    use_precomputed_states = (
        cache_params is not None and
        seq_len == 1 and
        cache_position is not None and
        cache_params.has_previous_state
    )
    # ===== TODO: KV Cache - Linear Attention 判断是否走 decode 单步路径 (END) =====


    # ===== TODO: KV Cache - Linear Attention 从缓存读取本层状态 (START) =====
    # 取出本层的卷积状态（滑动窗口）和递推状态（记忆矩阵），decode 路径会传给各自的函数
    conv_state = None
    recurrent_state = None
    if cache_params is not None:
        conv_state = cache_params.conv_states[layer_idx]
        recurrent_state = cache_params.recurrent_states[layer_idx]
        # 处理GQA情况：确保recurrent_state的头数与当前需要的头数匹配
        if recurrent_state is not None:
            # 计算当前需要的头数
            num_k_heads = linear_attn_module.num_k_heads
            num_v_heads = linear_attn_module.num_v_heads
            if num_v_heads > num_k_heads:
                # 计算重复次数
                repeat_factor = num_v_heads // num_k_heads
                # 沿头维度重复
                recurrent_state = recurrent_state.repeat_interleave(repeat_factor, dim=1)
    # ===== TODO: KV Cache - Linear Attention 从缓存读取本层状态 (END) =====

    # Q/K/V 拼接投影，转为 (batch, hidden, seq_len) 以备卷积
    mixed_qkv = linear_attn_module.in_proj_qkv(hidden_states).transpose(1, 2)
    # gating 向量 z，用于输出门控归一化
    z = linear_attn_module.in_proj_z(hidden_states)
    z = z.reshape(batch_size, seq_len, -1, linear_attn_module.head_v_dim)
    # 更新门 b（→ beta）和衰减门 a（→ g）的原始投影
    b = linear_attn_module.in_proj_b(hidden_states)
    a = linear_attn_module.in_proj_a(hidden_states)

    if use_precomputed_states:
        # Decode 路径：用单步卷积函数处理新 token，conv_state 在函数内部被原地更新
        mixed_qkv = torch_causal_conv1d_update(
            mixed_qkv,
            conv_state,
            linear_attn_module.conv1d.weight.squeeze(1),
            linear_attn_module.conv1d.bias,
            activation="silu",
        )
    else:
        # Prefill 路径：对整段序列做完整因果卷积（conv1d 内置左侧 causal padding），截取有效长度后激活
        # ===== TODO: KV Cache - Linear Attention 保存卷积状态到缓存 (START) =====
        # 用 F.pad(mixed_qkv, (left, 0)) 将 mixed_qkv 保存为滑动窗口快照，存入缓存对应层的槽位
        # left = 卷积核的最后一维 - mixed_qkv 的最后一维（正值左侧补零、负值从左裁剪，结果始终是 kernel_size 帧）
        if cache_params is not None:
            kernel_size = linear_attn_module.conv1d.weight.shape[-1]
            left = kernel_size - mixed_qkv.shape[-1]
            padded_state = F.pad(mixed_qkv, (left, 0))
            cache_params.conv_states[layer_idx] = padded_state
        # ===== TODO: KV Cache - Linear Attention 保存卷积状态到缓存 (END) =====
        mixed_qkv = F.silu(linear_attn_module.conv1d(mixed_qkv)[:, :, :seq_len])

    # 将 mixed_qkv 拆分为独立的 Q/K/V，各自 reshape 为多头格式
    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv,
        [linear_attn_module.key_dim, linear_attn_module.key_dim, linear_attn_module.value_dim],
        dim=-1,
    )
    query = query.reshape(batch_size, seq_len, -1, linear_attn_module.head_k_dim)
    key   = key.reshape(batch_size, seq_len, -1, linear_attn_module.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, linear_attn_module.head_v_dim)

    # beta：写入强度；g：遗忘因子（负值，由可学习参数 A_log 和动态偏置 dt_bias 决定）
    beta = b.sigmoid()
    g = -linear_attn_module.A_log.float().exp() * F.softplus(a.float() + linear_attn_module.dt_bias)
    # GQA：若 v_heads > k_heads，将 Q/K 按比例扩展以匹配 v_heads
    if linear_attn_module.num_v_heads // linear_attn_module.num_k_heads > 1:
        query = query.repeat_interleave(linear_attn_module.num_v_heads // linear_attn_module.num_k_heads, dim=2)
        key   = key.repeat_interleave(linear_attn_module.num_v_heads // linear_attn_module.num_k_heads, dim=2)

    if not use_precomputed_states:
        # Prefill：chunk-wise 并行 delta rule；output_final_state=True 时返回最终记忆矩阵
        core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(
            query, key, value,
            g=g, beta=beta,
            initial_state=None,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )
    else:
        # Decode：从缓存的记忆矩阵出发，单步递推；output_final_state=True 时返回更新后的矩阵
        core_attn_out, last_recurrent_state = torch_recurrent_gated_delta_rule(
            query, key, value,
            g=g, beta=beta,
            initial_state=recurrent_state,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )

    # ===== TODO: KV Cache - Linear Attention 保存递推状态到缓存 (START) =====
    # 将更新后的记忆矩阵写回缓存，供下一步 decode 使用
    # 处理GQA情况：只保存原始头数的recurrent_state
    if cache_params is not None and last_recurrent_state is not None:
        # 计算GQA分组数
        num_k_heads = linear_attn_module.num_k_heads
        if linear_attn_module.num_v_heads > num_k_heads:
            # 只保存第一个分组的状态，其他分组是重复的
            cache_params.recurrent_states[layer_idx] = last_recurrent_state[:, :num_k_heads]
        else:
            cache_params.recurrent_states[layer_idx] = last_recurrent_state
    # ===== TODO: KV Cache - Linear Attention 保存递推状态到缓存 (END) =====

    # 门控归一化：用 z 对输出做通道归一化，增强表达能力
    core_attn_out = core_attn_out.reshape(-1, linear_attn_module.head_v_dim)
    z = z.reshape(-1, linear_attn_module.head_v_dim)
    core_attn_out = linear_attn_module.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

    output = linear_attn_module.out_proj(core_attn_out)
    return output
