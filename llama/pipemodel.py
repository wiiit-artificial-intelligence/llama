# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import os
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from fairscale.nn import Pipe
from torch import nn
from torch.nn.parameter import Parameter


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048
    device: Optional[str] = 'cpu'
    tensor_parallel: Optional[bool] = False
    pipeline_parallel: Optional[bool] = True
    pipeline_chunks: Optional[int] = 1


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
        
        if args.tensor_parallel:
            model_parallel_size = fs_init.get_model_parallel_world_size()
        else:
            model_parallel_size = 1

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.device = args.device

        with torch.no_grad():
            self.wq_ = Parameter(torch.randn((args.n_heads * self.head_dim, args.dim)))
            self.wk_ = Parameter(torch.randn((self.n_kv_heads * self.head_dim, args.dim)))
            self.wv_ = Parameter(torch.randn((self.n_kv_heads * self.head_dim, args.dim)))
            self.wo_ = Parameter(torch.randn((args.n_heads * self.head_dim, args.dim)))

        if args.tensor_parallel:
            self.wq = ColumnParallelLinear(
                args.dim,
                args.n_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x,
                # init_method=lambda x: self.wq_
            )
            self.wq.weight = self.wq_

            self.wk = ColumnParallelLinear(
                args.dim,
                self.n_kv_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x,
                # init_method=lambda x: self.wk_
            )
            self.wk.weight = self.wk_

            self.wv = ColumnParallelLinear(
                args.dim,
                self.n_kv_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x,
                # init_method=lambda x: self.wv_
            )
            self.wv.weight = self.wv_

            self.wo = RowParallelLinear(
                args.n_heads * self.head_dim,
                args.dim,
                bias=False,
                input_is_parallel=True,
                init_method=lambda x: x,
                # init_method=lambda x: self.wo_
            )
            self.wo.weight = self.wo_
        else:
            self.wq = nn.Linear(
                args.dim,
                args.n_heads * self.head_dim,
                bias=False,)
            self.wq.weight = self.wq_
            # self.wq = lambda x: F.linear(x, weight=self.wq_, bias=None)

            self.wk = nn.Linear(
                args.dim,
                self.n_kv_heads * self.head_dim,
                bias=False,
            )
            self.wk.weight = self.wk_
            # self.wk = lambda x: F.linear(x, weight=self.wk_, bias=None)

            self.wv = nn.Linear(
                args.dim,
                self.n_kv_heads * self.head_dim,
                bias=False,
            )
            self.wv.weight = self.wv_
            # self.wv = lambda x: F.linear(x, weight=self.wv_, bias=None)

            self.wo = nn.Linear(
                args.n_heads * self.head_dim,
                args.dim,
                bias=False,
            )
            self.wo.weight = self.wo_
            # self.wo = lambda x: F.linear(x, weight=self.wo_, bias=None)

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

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

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
        args: Optional[ModelArgs],
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

        with torch.no_grad():
            self.w1_ = Parameter(torch.randn((hidden_dim, dim)))
            self.w2_ = Parameter(torch.randn((dim, hidden_dim)))
            self.w3_ = Parameter(torch.randn((hidden_dim, dim)))

        if args is not None and args.tensor_parallel == False:
            self.w1 = nn.Linear(dim, hidden_dim, bias=False)
            self.w1.weight = self.w1_
            # self.w1 = lambda x: F.linear(x, weight=self.w1_, bias=None)
            
            self.w2 = nn.Linear(hidden_dim, dim, bias=False)
            self.w2.weight = self.w2_
            # self.w2 = lambda x: F.linear(x, weight=self.w2_, bias=None)

            self.w3 = nn.Linear(dim, hidden_dim, bias=False)
            self.w3.weight = self.w3_
            # self.w3 = lambda x: F.linear(x, weight=self.w3_, bias=None)
        else:
            self.w1 = ColumnParallelLinear(
                dim, 
                hidden_dim, bias=False, 
                gather_output=False, 
                init_method=lambda x: x
                # init_method=lambda x: self.w1_
            )
            self.w1.weight = self.w1_

            self.w2 = RowParallelLinear(
                hidden_dim, 
                dim, bias=False, 
                input_is_parallel=True, 
                init_method=lambda x: x
                # init_method=lambda x: self.w2_
            )
            self.w2.weight = self.w2_

            self.w3 = ColumnParallelLinear(
                dim, 
                hidden_dim, 
                bias=False, 
                gather_output=False, 
                init_method=lambda x: x
                # init_method=lambda x: self.w3_
            )
            self.w3.weight = self.w3_
        

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
            args=args,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.start_pos = None
        self.freqs_cis = None
        self.mask = None

    def forward(
        self,
        x: torch.Tensor,
        start_pos: Optional[int]=None,
        freqs_cis: Optional[torch.Tensor]=None,
        mask: Optional[torch.Tensor]=None,
    ):
        start_pos = start_pos if start_pos is not None else self.start_pos
        freqs_cis = freqs_cis if freqs_cis is not None else self.freqs_cis
        mask = mask if mask is not None else self.mask
        return self.__forward(x, start_pos, freqs_cis, mask)

    def __forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
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
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        """
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.device = params.device

        with torch.no_grad():
            # self.tok_embeddings_ = Parameter(torch.randn(params.vocab_size, params.dim))
            self.tok_embeddings_ = Parameter(torch.Tensor(params.vocab_size, params.dim))
            self.tok_embeddings_ = nn.init.xavier_normal_(self.tok_embeddings_)

            self.output_ = Parameter(torch.randn(params.vocab_size, params.dim))

        if params is not None and params.tensor_parallel == False:
            # self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
            self.tok_embeddings = lambda x: F.embedding(x, weight=self.tok_embeddings_)
        else:
            self.tok_embeddings = ParallelEmbedding(
                params.vocab_size, 
                params.dim, 
                init_method=lambda x: x
            )
            self.tok_embeddings.weight = self.tok_embeddings_

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        if params is not None and params.tensor_parallel == False:
            self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
            self.output.weight = self.output_
            # self.output = lambda x: F.linear(x, weight=self.output_, bias=None)
        else:
            self.output = ColumnParallelLinear(
                params.dim, 
                params.vocab_size, 
                bias=False, 
                init_method=lambda x: x
            )
            self.output.weight = self.output_

        self.freqs_cis = precompute_freqs_cis(
            # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096. 
            # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

        self.sequential_layers = nn.Sequential(*self.layers, self.norm, self.output)

        if params.pipeline_parallel:
            world_size = int(os.environ["WORLD_SIZE"])
            chunks = self.params.pipeline_chunks

            # Transformer Block (layer) devices
            layer_devices = [torch.device(f"cpu:{cpu_id}") for cpu_id in range(world_size)]

            devices = layer_devices + [layer_devices[0]] # last partition executed in same device than first partition
            print(devices)

            # pipeline balance: (layers partition 0) (layers partition 1) ... (last layers partition) (norm + output)
            pipe_balance = [len(self.layers)//(world_size)] * (world_size)
            pipe_balance.append(2) #norm and output layer at last pipeline partition
            print(pipe_balance)

            self.sequential_layers = Pipe(
                self.sequential_layers,
                balance=pipe_balance, 
                chunks=chunks,
                checkpoint="never",
                devices=devices
                )
            print(self.sequential_layers)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            start_pos (int): Starting position for attention caching.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        with torch.no_grad():
            _bsz, seqlen = tokens.shape
            h = self.tok_embeddings(tokens)

            self.freqs_cis = self.freqs_cis.to(h.device)
            freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

            mask = None
            if seqlen > 1:
                mask = torch.full(
                    (1, 1, seqlen, seqlen), float("-inf"), device=tokens.device
                )
                mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

            for layer in self.layers:
                # h = layer(h, start_pos, freqs_cis, mask)
                layer.start_pos = start_pos
                layer.freqs_cis = freqs_cis
                layer.mask = mask
            
            if self.params.pipeline_parallel:
                output = self.sequential_layers(h).float()
            else:
                for layer in self.layers:
                    h = layer(h)
                h = self.norm(h)
                output = self.output(h).float()

        return output
