import math
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, List
from einops import rearrange

attn_impl: Literal["naive", "absorb"] = "naive"
world_size = 1 # number of nodes, The computation is running on a single device (no distributed training).
rank = 0 # The single process has a rank of 0 (since there is only one process).
block_size = 128 # The model processes sequences of up to 128 tokens at a time.

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


        MOE
        n_dense_layers (int): Number of dense layers in the model.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
    

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

     # MOE
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    moe_inter_dim: int = 1408 




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
            self.q_norm = torch.nn.RMSNorm(normalized_shape = self.q_lora_rank)
            self.wq_b = torch.nn.Linear(in_features= self.q_lora_rank, out_features=self.n_heads * self.qk_head_dim)
        
        self.wkv_a = torch.nn.Linear(in_features=self.dim, out_features=self.kv_lora_rank)
        self.kv_norm = torch.nn.RMSNorm(normalized_shape =self.kv_lora_rank)
        self.wkv_b = torch.nn.Linear(in_features=self.kv_lora_rank, out_features=self.n_heads * (self.qk_head_dim + self.v_head_dim))



        self.wo = torch.nn.Linear(in_features=self.n_heads * self.v_head_dim, out_features= self.dim)
        
        self.softmax_scale = self.qk_head_dim ** -0.5

        if attn_impl == "naive":
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_heads, self.v_head_dim), persistent=False)
        else:
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
       


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

        if attn_impl == 'naive':
            kv = self.wkv_b(self.kv_norm(kv)) # increase dimension
            kv = kv.view(bsz, seqlen, self.n_heads, self.qk_head_dim + self.v_head_dim) # batch, sequence lenght, number of heads, dimension
     
            k, v = torch.split(kv, [self.qk_head_dim, self.v_head_dim], dim=-1) # k^C, v^C
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            scores = torch.einsum("bshd,bthd->bsht", q, k[:bsz, :end_pos]) * self.softmax_scale
        else:
            wkv_b = self.wkv_b.weight
            wkv_b = wkv_b.view(self.n_heads, -1, self.kv_lora_rank)
            q = torch.einsum("bshd,hdc->bshc", q, wkv_b[:, :self.qk_head_dim])
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
            scores = (torch.einsum("bshc,btc->bsht", q, self.kv_cache[:bsz, :end_pos])) * self.softmax_scale

    
        
        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x) # score softmax of the scores and set the type as the same as the input x

                
        # x = torch.einsum("bsht,bthd->bshd", scores, v[:bsz, :end_pos])
       
        # x = self.wo(x.flatten(2))
        if attn_impl == "naive":
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        else:
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        x = self.wo(x.flatten(2))

        
       
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
    def __init__(self, dim: int, inter_dim: int):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x)) #Output=W_2(SiLU(W_1x) ⊙ (W_3x))



    
'''
    MOE implementation
'''
class Expert(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, dim: int, inter_dim: int):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, dim)
        self.w3 = nn.Linear(dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate
        n_routed_experts (int): Total number of routed experts.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        self.dim = getattr(args, 'dim', 128)
        self.topk = getattr(args, 'n_activated_experts', 6)  
        self.n_groups = getattr(args, 'n_expert_groups', 1)
        self.topk_groups = getattr(args, 'n_expert_groups', 1)  
        self.score_func = getattr(args, 'score_func', 1)   
        self.route_scale = getattr(args, 'route_scale', 1)  
        self.n_routed_experts = getattr(args, 'n_routed_experts', 1)  
        
        self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.bias = nn.Parameter(torch.empty(args.n_routed_experts)) if self.dim > 7168 else None #== 7168 else None 

        #stores indices and scores 
        self.expert_assignments = None  # To store expert indices during forward pass

       


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
                - weights: Tensor of shape (batch_size, topk) containing the gating weights.
                - indices: Tensor of shape (batch_size, topk) containing the indices of the selected experts.
        """
        # Compute affinity scores: s_i,t = u_t * e_i
        scores = F.linear(x, self.weight)    

        # Apply scoring function: s_i,t = softmax(scores) or sigmoid(scores)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores  # Store original scores for final weights
   
      
        if self.bias is not None:
            scores  = scores + self.bias  # s_i,t + b_i

        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(x), indices


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.dim = args.dim
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(args)
        self.experts = nn.ModuleList([
            Expert(args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
            for i in range(self.n_routed_experts)
        ])

      
        self.shared_experts = MLP(args.dim, args.n_shared_experts * args.moe_inter_dim)
         # keep expert assigmnet from route
        self.expert_assigment = None


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
      
  
       
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        
    
        z = self.shared_experts(x)

        self.expert_assigment = indices.view(shape[0], shape[1], self.n_activated_experts).detach().to("cpu")


        return  (y + z).view(shape) 

# ---------------------------------------------------------------------------------------------------------------------------------------------

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
        self.ffn = MLP(args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(args)
        self.attn_norm = nn.RMSNorm(args.dim)
        self.ffn_norm = nn.RMSNorm(args.dim)
        self.expert_assigment = None


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
        # x = x + self.attn(self.attn_norm(x), start_pos, mask)
        # x = self.ffn_norm(x)
        # x = x + self.ffn(x)
        # return x
        x = x + self.attn(self.attn_norm(x), start_pos, mask)
        x = x + self.ffn(self.ffn_norm(x))
    
        if isinstance(self.ffn, MoE):
            self.expert_assigment =  self.ffn.expert_assigment
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

    @torch.inference_mode()
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
        h = self.embed(tokens)

        
        for layer in self.layers:
            h = layer(h, start_pos, mask)
            
        
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
    
    def get_expert_assignments(self) -> List[Tuple[int, torch.Tensor]]:
        """
            It returns the layer and the number of activated experts based on expert
            example:
            (layer 1 , [count_exp1, count_exp2, count_exp3])
        """
        expert_asigment_by_layer = []
        
        for idx, layer in enumerate(self.layers):
            if isinstance(layer.ffn, MoE):
                assignments = layer.expert_assigment
                expert_counts = torch.bincount(assignments.flatten(), minlength=layer.ffn.n_routed_experts)
                expert_asigment_by_layer.append((idx, expert_counts))
        return expert_asigment_by_layer