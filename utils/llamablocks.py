import torch
from torch import nn
import torch.nn.functional as F

import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)

import time
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

PARALLEL_EMULATED = None

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    
        

    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

        

    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


@dataclass
class ModelArgs:
    name: str = 'LLaMa2-7b'
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048
    device: Optional[str] = 'cpu'
    parallel_emulated = PARALLEL_EMULATED

@dataclass
class ModelArgs7b:
    name: str = 'LLaMa2-7b'
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048
    device: Optional[str] = 'cpu'
    parallel_emulated = PARALLEL_EMULATED

@dataclass
class ModelArgs13b:
    name: str = 'LLaMa2-13b'
    dim: int = 5120
    n_layers: int = 40
    n_heads: int = 40
    n_kv_heads: Optional[int] = None
    vocab_size: int = 32000  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048
    device: Optional[str] = 'cpu'
    parallel_emulated = PARALLEL_EMULATED

@dataclass
class ModelArgs34b:
    name: str = 'LLaMa2-34b'
    dim: int = 8192
    n_layers: int = 48
    n_heads: int = 64
    n_kv_heads: Optional[int] = 8
    vocab_size: int = 32000  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = 1.0
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048
    device: Optional[str] = 'cpu'
    parallel_emulated = PARALLEL_EMULATED

@dataclass
class ModelArgs70b:
    name: str = 'LLaMa2-70b'
    dim: int = 8192
    n_layers: int = 80
    n_heads: int = 64
    n_kv_heads: Optional[int] = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 4096  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = 1.3
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048
    device: Optional[str] = 'cpu'
    parallel_emulated = PARALLEL_EMULATED

class Attention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, args: ModelArgs):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        if model_parallel_size == 1 and args.parallel_emulated is not None:
            model_parallel_size = args.parallel_emulated
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = max(self.n_kv_heads // model_parallel_size, 1)
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.device = args.device

        dim_parallelized = args.n_heads * self.head_dim if args.parallel_emulated is None \
            else self.n_local_heads * self.head_dim
        # print(f"dim_parallelized = {dim_parallelized}")
        
        kv_dim_parallelized = self.n_kv_heads * self.head_dim if args.parallel_emulated is None \
            else self.n_local_kv_heads * self.head_dim
        # print(f"kv_dim_parallelized = {kv_dim_parallelized}")

        self.wq = ColumnParallelLinear(
            args.dim,
            dim_parallelized,
            bias=False,
            gather_output=False,
            # init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            kv_dim_parallelized,
            bias=False,
            gather_output=False,
            # init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            kv_dim_parallelized,
            bias=False,
            gather_output=False,
            # init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            dim_parallelized,
            args.dim,
            bias=False,
            input_is_parallel=True,
            # init_method=lambda x: x,
        )

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).to(device=self.device)
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).to(device=self.device)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # print(xq.shape, xk.shape, xv.shape)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]
        # print(xq.shape, keys.shape, values.shape)
        # print(keys[0,0,:,0])

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        # print(xq.shape, keys.shape, values.shape)
        # print(keys[0,0,:,0])

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)
    
class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        parallel_emulated=None,
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.hidden_dim = hidden_dim

        hidden_dim = hidden_dim if parallel_emulated is None else hidden_dim // parallel_emulated

        self.w1 = ColumnParallelLinear(
            dim, 
            hidden_dim, 
            bias=False, 
            gather_output=False, 
            # init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, 
            dim, 
            bias=False, 
            input_is_parallel=True, 
            # init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, 
            hidden_dim, 
            bias=False, 
            gather_output=False, 
            # init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            parallel_emulated=args.parallel_emulated
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    @torch.inference_mode()
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        verbose=0
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        # Attention
        start_time = time.time()
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_cis, mask)
        stop_time = time.time()

        attention_time = stop_time - start_time

        # Feed-Forward
        start_time = time.time()
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        stop_time = time.time()

        feedforward_time = stop_time - start_time

        if verbose > 0:
            print(f"Attention time    = {attention_time*1e3:3f}[ms]")
            print(f"Feed-Forward time = {feedforward_time*1e3:3f}[ms]")
        
        return out

def feedforward_dim(args):
    hidden_dim = 4*args.dim
    hidden_dim = int(2 * hidden_dim / 3)
    # custom dim factor multiplier
    if args.ffn_dim_multiplier is not None:
        hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
    hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
    return hidden_dim

def theorical_layer_time(
        args, 
        n_workers, 
        worker_flops, 
        memory_bw, 
        network_bw, 
        network_latency, 
        dtype,
        batch_size,
        seq_ini_len,
        loop_j):
    F = feedforward_dim(args)
    D = args.dim
    H = args.n_heads
    Dh = D/H
    c = F / D
    b = dtype.itemsize * 8
    P = n_workers
    Li = seq_ini_len
    j = loop_j
    B = batch_size
    Pflops = worker_flops
    Tm = memory_bw
    Tl = network_latency
    Tbw = network_bw
    
    if D == 8192:
        o1, o2 = 2*(2.25 + 3*c), 3.25 + 3*c
    else:
        o1, o2 = 2*(4 + 3*c), 4 + 3*c

    # print(b, c, F, Dh, o1, o2)

    if j == 0:
        # first loop
        t_ops = (B*Li*D/Pflops) * (o1*D/P + Li*(5/Dh + 4/P) + 2)
        t_mem = (B*D*b/Tm) * (o2*D/P + (Li/P)*(5*P + 4 + c) + (Li**2)/(P*Dh))
        # t_mem = (B*D*b/Tm) * (o2*D/P + (Li/P)*(5*P + 4 + c) + (Li**2)/(P*Dh)) + 2*(B*b/Tm)*(Li**2)*(H/P) + 2*B*Li*b*D/Tm + \
        #     (B*b/Tm)*(3*Li*D/P + (H/P)*Li**2 + (H/P)*Li*Dh + 2*Li*D + 2*Li*D*c/P)
        if Tbw is None or P == 1:
            t_net = 0
        else:
            t_net = 4*(P-1) * (Tl + B*Li*D*b/(Tbw*P))
    else:
        # next j loop
        t_ops = (B*D/Pflops) * (o1*D/P + (Li + j)*(5/Dh + 4/P) + 2)
        t_mem = (B*D*b/Tm) * (o2*D/P + (1/P)*(5*P + 2 + c + 2*(Li+j)))
        if Tbw is None or P == 1:
            t_net = 0
        else:
            t_net = 4*(P-1) * (Tl + B*D*b/(Tbw*P))

    # print(f"t ops = {t_ops*1e3:.3f} - t mem = {t_mem*1e3:.3f} - t net = {t_net*1e3:.3f}")
    t_layer = t_ops + t_mem + t_net

    return t_layer, t_ops, t_mem, t_net

def get_model_args(name):
    class_mapper = {
        "llama2-7b": ModelArgs7b,
        "llama2-13b": ModelArgs13b,
        "llama2-34b": ModelArgs34b,
        "llama2-70b": ModelArgs70b
    }

    if name in class_mapper.keys():
        return class_mapper[name]()
    else:
        raise ValueError(f"Model name {name} is unknown!")

def optimal_num_workers(
        args, 
        worker_flops, 
        memory_bw, 
        network_bw, 
        network_latency, 
        dtype,
        batch_size,
        seq_ini_len,
        loop_j):
    F = feedforward_dim(args)
    D = args.dim
    H = args.n_heads
    Dh = D/H
    c = F / D
    b = dtype.itemsize * 8
    Li = seq_ini_len
    j = loop_j
    B = batch_size
    Pflops = worker_flops
    Tm = memory_bw
    Tl = network_latency
    Tbw = network_bw
    
    if D == 8192:
        o1, o2 = 2*(2.25 + 3*c), 3.25 + 3*c
    else:
        o1, o2 = 2*(4 + 3*c), 4 + 3*c

    if Tl is None or Tl == 0.0:
        P = H # maximum parallelization
        return int(P)

    if j == 0:
        # first loop
        P_squared = (B*D/(4*Tl)) * ( (Li/Pflops)*(o1*D + 4*Li) + (b/Tm)*(o2*D + Li*(4+c) + (Li**2)/Dh) - 4*Li*b/Tbw )  
    else:
        # next j loop
        P_squared = (B*D/(4*Tl)) * ( (1/Pflops)*(o1*D + 4*(Li+j)) + (b/Tm)*(o2*D + 2*(Li+j)) - 4*b/Tbw )

    P_squared = max(1.0, P_squared)
    P = np.sqrt(P_squared)
    P = 2**np.round(np.log2(P))
    P = min(H, P)
    
    return int(P)
