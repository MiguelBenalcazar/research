import math
import torch
from torch import nn
from dataclasses import dataclass
from typing import Literal, Optional
from einops import rearrange

import sys
import os


path_DYT = os.path.join(os.getcwd(), "models")
sys.path.append(path_DYT)
from dynamicTanh import DyT

@dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        n_layers (int): Number of transformer layers.
        n_heads (int): Number of attention heads.
        input_dim (int): Number of input dimension
        
        Latent Space
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings. size of the future predictions
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.

        dropout (float): dropout value for training
    """
    max_batch_size: int = 8
    max_seq_len: int = 128
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 10
    dim: int = 128
    inter_dim: int = 256
    n_layers: int = 1
    n_heads: int = 1
    input_dim:int = 10

    q_lora_rank: int = 64
    kv_lora_rank: int = 64
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    dropout: float=0.5


class MLA(nn.Module):
    """
    Multi-Headed Attention Layer (MLA).

    Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_head_dim (int): Total dimensionality of query/key projections. 
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_head_dim = args.qk_nope_head_dim 
        self.v_head_dim = args.v_head_dim

        # Query
        if self.q_lora_rank == 0:
            self.wq = torch.nn.Linear(in_features=self.dim, out_features= self.n_heads * self.qk_head_dim) # normal Query (no latent space)
        else:
            self.wq_a = torch.nn.Linear(in_features = self.dim, out_features = self.q_lora_rank)  # latent space (Q reduced )
            self.q_norm = DyT(num_features = self.q_lora_rank , alpha_init_value=0.5)
            self.wq_b = torch.nn.Linear(in_features= self.q_lora_rank, out_features=self.n_heads * self.qk_head_dim)
        
        self.wkv_a = torch.nn.Linear(in_features=self.dim, out_features=self.kv_lora_rank)
        self.kv_norm = DyT(num_features =self.kv_lora_rank, alpha_init_value=0.5)
        self.wkv_b = torch.nn.Linear(in_features=self.kv_lora_rank, out_features=self.n_heads * (self.qk_head_dim + self.v_head_dim))



        self.wo = torch.nn.Linear(in_features=self.n_heads * self.v_head_dim, out_features= self.dim)
        
        self.softmax_scale = self.qk_head_dim ** -0.5

        # Dropout layer
        self.dropout = nn.Dropout(args.dropout)

        self._reset_parameters()
        # include K,Q,V
    
    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation

        if self.q_lora_rank == 0:
            nn.init.xavier_uniform_(self.wq.weight)
            self.wq.bias.data.fill_(0)
        else:
            nn.init.xavier_uniform_(self.wq_a.weight)
            self.wq_a.bias.data.fill_(0)

            nn.init.xavier_uniform_(self.wq_b.weight)
            self.wq_b.bias.data.fill_(0)
        
        nn.init.xavier_uniform_(self.wkv_a.weight)
        self.wkv_a.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.wkv_b.weight)
        self.wkv_b.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.wo.weight)
        self.wo.bias.data.fill_(0)

    def forward(self, x: torch.Tensor, start_pos: int,  mask: Optional[torch.Tensor],  return_attention: bool= False):
        """
        Forward pass for the Multi-Headed Attention Layer (MLA).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
 
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        
        # Queries
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            # reduction (latent space) -> norm -> increse space 
            q = self.wq_b(self.q_norm(self.wq_a(x)))

        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim) # batch, sequence lenght, number of heads, dimension
        
        # KV -> keys , values
        kv = self.wkv_a(x) # get kv reduction
     
        kv = self.wkv_b(self.kv_norm(kv)) # increase dimension
        kv = kv.view(bsz, seqlen, self.n_heads, self.qk_head_dim + self.v_head_dim) # batch, sequence lenght, number of heads, dimension
     
        k, v = torch.split(kv, [self.qk_head_dim, self.v_head_dim], dim=-1) # k^C, v^C
            
        scores = torch.einsum("bshd,bthd->bsht", q, k[:bsz, :end_pos]) * self.softmax_scale
        
        
        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x) # score softmax of the scores and set the type as the same as the input x


        # Apply Dropout on the attention scores
        scores = self.dropout(scores)
                
        x = torch.einsum("bsht,bthd->bshd", scores, v[:bsz, :end_pos])
       
        x = self.wo(x.flatten(2))

        # Apply Dropout to the final output before returning
        x = self.dropout(x)
        
       
        if return_attention:
            return x, scores
        else:
            return x
        

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int, dropout_prob: float = 0.1):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        
        self.w1 = nn.Linear(in_features=dim, out_features=inter_dim)
        self.w2 = nn.Linear(in_features=inter_dim, out_features=dim)
        self.w3 = nn.Linear(in_features=dim, out_features=inter_dim)

        self.dropout = nn.Dropout(dropout_prob)  # Dropout layer with specified probability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        x1 = self.dropout(self.w1(x))
        x3 = self.dropout(self.w3(x))

        return self.dropout(self.w2(nn.functional.silu(x1) * x3)) #Output=W_2(SiLU(W_1x) ⊙ (W_3x))


class Block(nn.Module):
    """
    Transformer block combining attention and feed-forward layers.

    Attributes:
        attn (nn.Module): Attention layer (MLA).
        ffn (nn.Module): Feed-forward network (MLP or MoE).
        attn_norm (nn.Module): Layer normalization for attention.
        ffn_norm (nn.Module): Layer normalization for feed-forward network.
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initializes the Transformer block.

        Args:
            layer_id (int): Layer index in the transformer.
            args (ModelArgs): Model arguments containing block parameters.
        """
        super().__init__()
        self.attn = MLA(args)
        self.ffn = MLP(args.dim, args.inter_dim) #if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = DyT(num_features = args.dim, alpha_init_value=0.5)
        self.ffn_norm = DyT(num_features =args.dim, alpha_init_value=0.5)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor, start_pos: int, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position in the sequence.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """
        x = x + self.dropout(self.attn(self.attn_norm(x), start_pos, mask))
        x = self.ffn_norm(x)
        x = x + self.dropout(self.ffn(x))
        return x
    
class Transformer(nn.Module):
    """
    Transformer model with positional embeddings, multiple layers, and output projection.

    Attributes:
        max_seq_len (int): Maximum sequence length for the transformer.
        embed (nn.Module): Embedding layer for input tokens.
        layers (nn.ModuleList): List of transformer blocks.
        norm (nn.Module): Layer normalization applied after all blocks.
        head (nn.Module): Output projection layer mapping to vocabulary size.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the Transformer model.

        Args:
            args (ModelArgs): Model arguments containing transformer parameters.
        """
        super().__init__()
        self.max_seq_len = args.max_seq_len
        # self.embed = nn.Linear(args.vocab_size, args.dim)

        
        self.embed = nn.Linear(in_features = args.input_dim, out_features= args.dim)
        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args))

        self.dropout = nn.Dropout(args.dropout)  # Dropout for embeddings

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        
        nn.init.xavier_uniform_(self.embed.weight)
        self.embed.bias.data.fill_(0)
       


    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor], start_pos: int = 0):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
            start_pos (int, optional): Starting position in the sequence for rotary embeddings. Defaults to 0.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """

        
        seqlen = tokens.size(1)  
        h = self.dropout(self.embed(tokens))

        
        for layer in self.layers:
            h = layer(h, start_pos, mask)
            h = self.dropout(h)
        return h
    

    def get_attention_maps(self, tokens: torch.Tensor, mask=Optional[torch.Tensor], start_pos: int = 0):
        seqlen = tokens.size(1)
        h = self.embed(tokens)
       
        attention_maps = []
        for layer in self.layers:
            _, attn_map = layer.attn(x=h, start_pos = start_pos, mask = mask, return_attention = True)
            attention_maps.append(rearrange(attn_map, "b s h d -> b h s d"))
            h = layer(h, start_pos,  mask)
        return attention_maps