import math
from typing import List, Optional, Tuple, Union  # noqa: F401

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from rich.traceback import install
from transformers import (  # noqa: F401
    GenerationMixin,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast  # noqa: F401

install()


# 配置：MokioMindConfig（相当于“模型的说明书”）
# 用于存放所有超参数，就是一个配置文件可以类比游戏中角色属性的设定
class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: Optional[int] = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        # MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.1,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 4,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )


# 功能类似 LayerNorm，但计算方式稍不同（基于均方根, RMS），用来稳定训练。
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return x * self.weight * self._norm(x).float().type_as(x)


# 目的是给注意力里的 Query 和 Key 加上“位置信息”，让模型分辨“前后顺序”。
# 这里用的方式是 RoPE（rotary positional embeddings）。
def precompute_freqs(
    dim: int,
    end: int = (32 * 1024),
    rope_base: float = 1e6,
    rope_scaling: Optional[dict] = None,
):
    # 写出最初的RoPE式子
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # 根据公式计算
    if rope_scaling is not None:
        orig_max = rope_scaling.get("original_max_position_embeddings", 2048)
        factor = rope_scaling.get("factor", 4)
        beta_fast = rope_scaling.get("beta_fast", 4)
        beta_slow = rope_scaling.get("beta_slow", 1)
        # else:
        #     orig_max = 2048
        #     factor = 4
        #     beta_fast = 4
        #     beta_slow = 1

        # 计算corr_dim
        corr_dim = next(
            (i for i in range(1, dim // 2) if 2 * math.pi / freqs[i] >= orig_max),
            dim // 2,
        )

        # 计算power
        power = torch.arange(0, dim // 2, device=freqs.device).float() / (
            max(dim // 2 - 1, 1)
        )

        # 计算beta
        beta = beta_fast + power * (beta_slow - beta_fast)

        # 计算scale
        scale = torch.where(
            torch.arange(dim // 2, device=freqs.device) < corr_dim,
            (beta * factor - beta + 1) / (beta * factor),
            1.0 / factor,
        )

        # 计算缩放后的freqs
        freqs = freqs * scale
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embedded = (q * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(q) * sin.unsqueeze(unsqueeze_dim)
    )
    k_embedded = (k * cos.unsqueeze(unsqueeze_dim)) + (
        rotate_half(k) * sin.unsqueeze(unsqueeze_dim)
    )
    return q_embedded, k_embedded


# 用来把少量的 key/value 头（KV）复用到更多的 query 头（Q）
def repeat_kv(x, n_rep: int):
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    if n_rep == 1:
        return x
    bs, slen, key_value_heads, dim = x.shape
    x = x[:, :, :, None, :].expand(bs, slen, key_value_heads, n_rep, dim)
    x = x.reshape(bs, slen, key_value_heads * n_rep, dim)
    return x


"""
假设输入 x.shape = (2, 10, 4, 64)（bs=2，slen=10，key_value_heads=4，dim=64），n_rep=2：
插入新维度后：(2, 10, 4, 1, 64)；
expand 后：(2, 10, 4, 2, 64)（每个键值头复制 2 份）；
reshape 后：(2, 10, 8, 64)（4*2=8 个扩展后的键值头）。
"""

"""
        假设外部参数为：
        args.num_attention_heads = 16（16 个 Q 头）
        self.num_key_value_heads = 8（8 个 KV 头）
        args.hidden_size = 1024（总隐藏层维度）
        则计算结果：
        n_local_heads = 16
        n_local_kv_heads = 8
        n_rep = 16 // 8 = 2（每个 KV 头对应 2 个 Q 头）
        head_dim = 1024 // 16 = 64（每个 Q/KV 头的维度都是 64）
        最终：16 个 Q 头（每个 64 维）、8 个 KV 头（每个 64 维，复用 2 次），
        总维度均为 16×64=1024，与隐藏层维度一致。

"""


class Attention(nn.Module):
    def __init__(self, args: MokioMindConfig):
        super().__init__()

        self.num_key_value_heads = (
            args.num_attention_heads
            if args.num_key_value_heads is None
            else args.num_key_value_heads
        )

        assert args.num_attention_heads % self.num_key_value_heads == 0

        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        # 隐藏层
        self.q_proj = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            args.hidden_size, args.num_attention_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim, args.hidden_size, bias=False
        )

        # flash and dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
            and args.flash_attention
        )

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        bsz, seq_len, _ = x.shape

        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)

        past_kv = (xk, xv) if use_cache else None

        xq = xq.transpose(1, 2)
        # [bsz, n_local_heads, seq_len, hidden_dim]
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        if (
            self.flash
            and seq_len > 1
            and (attention_mask is None or torch.all(attention_mask == 1))
        ):
            attn_mask = (
                None
                if attention_mask is None
                else attention_mask.view(bsz, 1, 1, -1)
                .expand(bsz, self.n_local_heads, seq_len, -1)
                .bool()
            )
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,  # 自回归（因果）注意力
            )
        else:

            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=-1,
            )

            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)

        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor):
        gated = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.dropout(self.down_proj(gated))


class MoEGate(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(
            torch.empty((self.config.n_routed_experts, self.gating_dim))
        )
        self.reset_parameters()
        self.n_routed_experts = self.config.n_routed_experts

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)

        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(
                f"insupportable scoring function for MoE gating: {self.scoring_func}"
            )

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(
                    bsz, self.n_routed_experts, device=hidden_states.device
                )
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device),
                )
                ce = ce.div(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(
                    dim=1
                ).mean() * self.alpha
            else:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts
                )
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0
        return topk_weight, topk_idx, aux_loss


class MoEFeedForward(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.config = config
        # 专家层
        self.experts = nn.ModuleList([
            FeedForward(config) for _ in range(config.n_routed_experts)
        ])  # 所有专家是独立初始化


class MoEFeedForaward(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.config = config
        # 专家层
        self.experts = nn.ModuleList([
            FeedForward(config) for _ in range(config.n_routed_experts)
        ])
        # 门控层
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config) for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, h = orig_shape

        # 使用门控机制旋转专家
        topk_weight, topk_idx, aux_loss = self.gate(x)
        # 展开x以便处理
        x = x.view(-1, x.shape[-1])

        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            # 按照定义的num_experts_per_tok重复输入token
            # 每个token安排num_experts_per_tok个专家处理
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            # y是空张量，和x形状相同
            y = torch.empty_like(x, dtype=torch.float32)
            # 遍历所有专家
            for i, expert in enumerate(self.experts):
                # 找到所有指向专家i的token
                # 然后将这些token输入专家i进行处理
                # 最后将结果放回y对应位置
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)
            # 加权求和
            # 最后的y意义是每个token经过专家处理后的加权结果
            y = y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1).sum(dim=1)
            y = y.view(*orig_shape)
        # 如果是推理阶段
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(
                *orig_shape
            )
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    # MoE推理方法
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        # 使用cache，创建一个和x形状相同的零张量
        expert_cache = torch.zeros_like(x)
        # 对专家索引进行排序，最后是[0,0,0,1,1,2,2,2,...]这样的顺序
        # 分拣
        idxs = flat_expert_indices.argsort()
        # 统计每个专家被分配到的token数量
        # 打包
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # 计算每个token对应的专家索引
        token_idxs = idxs // self.config.num_experts_per_tok
        # 对每个打包好的包进行处理
        for i, end_idx in enumerate(tokens_per_expert):
            # 计算当前包的起始位置
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            # 取出当前包对应的专家
            expert = self.experts[i]
            # 取出token对应的原始id
            exp_token_idx = token_idxs[start_idx:end_idx]
            # 取出token对应的数据
            expert_tokens = x[exp_token_idx]
            # 计算专家输出，一次性处理当前包的所有token
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 加权
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 将结果散点加到缓存中对应位置
            expert_cache.scatter_add_(
                0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out
            )

        return expert_cache


class MokioMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MokioMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attention = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.ffn = (
            FeedForward(config) if not config.use_moe else MoEFeedForaward(config)
        )

    def forward(
        self,
        hidden_states,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        res = hidden_states

        hidden_states, present_key_value = self.self_attention(
            self.input_layernorm(hidden_states),  # pre-norm
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )

        hidden_states = res + hidden_states

        hidden_states = hidden_states + self.ffn(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, present_key_value


class MokioMindModel(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = (
            config.vocab_size,
            config.num_hidden_layers,
        )
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            MokioMindBlock(l, config) for l in range(self.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs(
            dim=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs,
    ):
        # input_ids: [bsz, seq_len]
        if input_ids is None:
            raise ValueError("input_ids cannot be None")
        batch_size, seq_length = input_ids.shape

        if hasattr(past_key_values, "layers"):
            past_key_values = None

        past_key_values = past_key_values or [(None, None)] * len(self.layers)

        # 计算start_pos：如果存在past，则start_pos为已有past序列长度
        if (
            past_key_values
            and past_key_values[0] is not None
            and past_key_values[0][0] is not None
        ):
            # past_key_values[0][0] 通常是 key 张量，形状类似 [batch, seq_len, head_dim]
            start_pos = past_key_values[0][0].shape[1]
        else:
            start_pos = 0

        # Embedding + dropout
        hidden_states = self.dropout(
            self.embed_tokens(input_ids)
        )  # [bsz, seq_len, hidden]

        position_embeddings = (
            self.freqs_cos[start_pos : start_pos + seq_length],
            self.freqs_sin[start_pos : start_pos + seq_length],
        )
        presents = []
        # Transformer 的每一层（比如 Layer 1、Layer 2…Layer k），
        # 都会单独初始化一组 Q、K、V 的线性变换矩阵（即每个层都有自己的W_Q、W_K、W_V）。

        # 这些矩阵是模型在训练时 “逐层学习” 的：
        # 不同层需要捕捉不同层级的语义信息（比如底层学字面关联，高层学逻辑结构），
        # 所以需要独立的参数来适配对应层级的计算。
        for layer_idx, (layer, past_key_value) in enumerate(
            zip(self.layers, past_key_values)
        ):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        return hidden_states, presents


class MokioMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MokioMindConfig

    def __init__(self, config: MokioMindConfig):
        super().__init__(config)
        self.model = MokioMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **args,
    ):
        h, past_kvs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args,
        )

        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(h[:, slice_indices, :])

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_kvs,
            hidden_states=h,
        )
