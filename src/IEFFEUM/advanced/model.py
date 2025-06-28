import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
import math
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from inspect import isfunction

# highly inspired by the following repositories:
# https://github.com/lucidrains/alphafold2
# https://github.com/microsoft/unilm/tree/master/Diff-Transformer

# Helper Functions
def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)

def symmetrize_tensor(x):
    # x: shape [b,L,L,dim]
    # Symmetrize along the last two dimensions
    return 0.5 * (x + x.transpose(1, 2))

# Feed Forward Classes
class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 2,
        dropout = 0.,
        out_dim = None
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, out_dim if out_dim else dim)
        )
        init_zero_(self.net[-1])
        
    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.net(x)

# Attention Classes
class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True))
        x_norm = x / (rms + self.eps)
        return self.gamma * x_norm

class DiffAttn2d(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads=4,
        dim_head=8,
        dropout=0.,
    ):
        super().__init__()
        inner_dim = dim_head * num_heads * 2
        self.num_heads = num_heads
        self.scale = dim_head ** -0.5
        
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        
        self.gating = nn.Linear(dim, inner_dim)
        nn.init.constant_(self.gating.weight, 0.)
        nn.init.constant_(self.gating.bias, 1.)
        
        self.dropout = nn.Dropout(dropout)
        init_zero_(self.to_out)
        
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(dim_head, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(dim_head, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(dim_head, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(dim_head, dtype=torch.float32).normal_(mean=0, std=0.1))
        
        self.norm = nn.LayerNorm(2 * dim_head)

    def forward(self, x, mask=None, attn_bias=None, context=None, context_mask=None):
        device, h = x.device, self.num_heads
        has_context = exists(context)
        context = default(context, x)
        
        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        
        q, k = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=2*h), (q, k))
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)
        
        q = q * self.scale
        
        dots = einsum('b h i d, b h j d -> b h i j', q, k)
        
        mask = default(mask, lambda: torch.ones(1, q.shape[-2], device=device).bool())
        context_mask = mask if not has_context else default(context_mask, lambda: torch.ones(1, k.shape[-2], device=device).bool())
        mask_value = -torch.finfo(dots.dtype).max
        mask = mask[:, None, :, None] * context_mask[:, None, None, :]
        dots = dots.masked_fill(~mask, mask_value)
    
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        
        attn = dots.softmax(dim=-1)
        
        attn = attn.view(attn.shape[0], self.num_heads, 2, *attn.shape[2:])
        attn = attn[:, :, 0] - lambda_full * attn[:, :, 1]
        
        if exists(attn_bias):
            attn = attn + attn_bias.softmax(dim=-1)
            attn = self.dropout(attn)
        
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        
        out = self.norm(out)
        out = out * (1 - self.lambda_init)
        
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        gates = self.gating(x)
        out = out * gates.sigmoid()
        
        out = self.to_out(out)
        return out

class DiffAttn3d(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        depth=1,
        num_heads=4,
        dim_head=4,
    ):
        super().__init__()
        self.embed_dim = num_heads * dim_head * 2
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.out_dim = out_dim
        self.scaling = dim_head ** -0.5
        
        self.q = nn.Linear(in_dim, self.embed_dim, bias=False)
        self.kv = nn.Linear(in_dim, self.embed_dim * 2, bias=False)
        self.out = nn.Linear(self.embed_dim, out_dim, bias=False)
        
        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * (depth + 1))
        self.lambda_q1 = nn.Parameter(torch.zeros(self.dim_head, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.dim_head, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.dim_head, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.dim_head, dtype=torch.float32).normal_(mean=0, std=0.1))
        
        self.subln = RMSNorm(2 * self.dim_head)
    
    def forward(self, x, mask_2d):
        bsz, _, tgt_len, embed_dim = x.size()
        x = rearrange(x, 'b 1 l d -> b l d', b=bsz, l=tgt_len, d=embed_dim)
        
        q, k, v = (self.q(x), *self.kv(x).chunk(2, dim=-1))
        
        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.dim_head)
        k = k.view(bsz, tgt_len, 2 * self.num_heads, self.dim_head)
        v = v.view(bsz, tgt_len, self.num_heads, 2 * self.dim_head)
        
        q = rearrange(q, 'b l n d -> b n l d')
        k = rearrange(k, 'b l n d -> b n d l')
        v = rearrange(v, 'b n l d -> b l n d')
        
        q *= self.scaling
        dots = torch.matmul(q, k)
        mask_value = -torch.finfo(dots.dtype).max
        dots = dots.masked_fill(~repeat(mask_2d, 'b l i -> b n l i', n=2 * self.num_heads), mask_value)
        
        attn_weights = F.softmax(dots, dim=-1, dtype=torch.float32).type_as(dots)
        
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, tgt_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
        
        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.dim_head)
        
        return rearrange(self.out(attn), 'b l d -> b 1 l d', b=bsz, l=tgt_len, d=self.out_dim)

class AxialDiffAttn(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        row_attn = True,
        col_attn = True,
        accept_edges = False,
        **kwargs
    ):
        super().__init__()
        assert not (not row_attn and not col_attn), 'row or column attention must be turned on'
        
        self.row_attn = row_attn
        self.col_attn = col_attn
        
        self.norm = nn.LayerNorm(dim)
        
        self.attn = DiffAttn2d(dim = dim, num_heads = num_heads, **kwargs)
        
        self.edges_to_attn_bias = nn.Sequential(
            nn.Linear(dim, num_heads, bias = False),
            Rearrange('b i j h -> b h i j')
        ) if accept_edges else None
        
    def forward(self, x, edges = None, mask = None):
        assert self.row_attn ^ self.col_attn, 'has to be either row or column attention, but not both'
        
        b, h, w, d = x.shape
        
        x = self.norm(x)
        
        if self.col_attn:
            axial_dim = w
            mask_fold_axial_eq = 'b h w -> (b w) h'
            input_fold_eq = 'b h w d -> (b w) h d'
            output_fold_eq = '(b w) h d -> b h w d'
            
        elif self.row_attn:
            axial_dim = h
            mask_fold_axial_eq = 'b h w -> (b h) w'
            input_fold_eq = 'b h w d -> (b h) w d'
            output_fold_eq = '(b h) w d -> b h w d'
            
        x = rearrange(x, input_fold_eq)
        
        if exists(mask):
            mask = rearrange(mask, mask_fold_axial_eq)
            
        attn_bias = None
        if exists(self.edges_to_attn_bias) and exists(edges):
            attn_bias = self.edges_to_attn_bias(edges)
            attn_bias = repeat(attn_bias, 'b h i j -> (b x) h i j', x = axial_dim)
        
        out = self.attn(x, mask = mask, attn_bias = attn_bias,)
        out = rearrange(out, output_fold_eq, h = h, w = w)
        
        return out

# Triangular Multiplicative Update
class TriangleMultiplicativeModule(nn.Module):
    def __init__(
        self,
        *,
        dim,
        hidden_dim = None,
        mix = 'ingoing'
    ):
        super().__init__()
        assert mix in {'ingoing', 'outgoing'}, 'mix must be either ingoing or outgoing'
        
        hidden_dim = default(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)
        
        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)
        
        self.left_gate = nn.Linear(dim, hidden_dim)
        self.right_gate = nn.Linear(dim, hidden_dim)
        self.out_gate = nn.Linear(dim, hidden_dim)
        
        for gate in (self.left_gate, self.right_gate, self.out_gate):
            nn.init.constant_(gate.weight, 0.)
            nn.init.constant_(gate.bias, 1.)
            
        if mix == 'outgoing':
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        elif mix == 'ingoing':
            self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'
            
        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim)
        
    def forward(self, x, mask = None):
        assert x.shape[1] == x.shape[2], 'feature map must be symmetrical'
        if exists(mask):
            mask = rearrange(mask, 'b i j -> b i j ()')
            
        x = self.norm(x)
        
        left = self.left_proj(x)
        right = self.right_proj(x)
        
        if exists(mask):
            left = left * mask
            right = right * mask
            
        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()
        
        left = left * left_gate
        right = right * right_gate
        
        out = einsum(self.mix_einsum_eq, left, right)
        
        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)

# Evformer Blocks
class OuterMean(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim = None,
        eps = 1e-5
    ):
        super().__init__()
        self.eps = eps
        self.norm = nn.LayerNorm(dim)
        hidden_dim = default(hidden_dim, dim)
        
        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, dim)
        
    def forward(self, x, mask = None):
        x = self.norm(x)
        left = self.left_proj(x)
        right = self.right_proj(x)
        outer = rearrange(left, 'b m i d -> b m i () d') * rearrange(right, 'b m j d -> b m () j d')
        
        if exists(mask):
            mask = rearrange(mask, 'b m i -> b m i () ()') * rearrange(mask, 'b m j -> b m () j ()')
            outer = outer.masked_fill(~mask, 0.)
            outer = outer.mean(dim = 1) / (mask.sum(dim = 1) + self.eps)
        else:
            outer = outer.mean(dim = 1)
            
        return self.proj_out(outer)

class TriangleMultiplicativeBlock(nn.Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.outer_mean = OuterMean(dim)
        self.triangle_multiply_outgoing = TriangleMultiplicativeModule(dim = dim, mix = 'outgoing')
        self.triangle_multiply_ingoing = TriangleMultiplicativeModule(dim = dim, mix = 'ingoing')
        
    def forward(
        self,
        x,
        mask = None,
        s = None,
        seq_mask = None
    ):
        if exists(s):
            x = x + self.outer_mean(s, mask = seq_mask)
        x = self.triangle_multiply_outgoing(x, mask = mask) + x
        x = self.triangle_multiply_ingoing(x, mask = mask) + x
        
        return x

class SeqAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        dim_head,
        dropout = 0.,
        depth = None
    ):
        super().__init__()
        self.row_attn = AxialDiffAttn(dim = dim, num_heads = num_heads, dim_head = dim_head, row_attn = True, col_attn = False, accept_edges = True, dropout = dropout, depth = depth)
        self.col_attn = AxialDiffAttn(dim = dim, num_heads = num_heads, dim_head = dim_head, row_attn = False, col_attn = True, accept_edges = False, dropout = dropout, depth = depth)

    def forward(
        self,
        x,
        mask = None,
        pairwise_repr = None
    ):
        x = self.row_attn(x, mask = mask, edges = pairwise_repr) + x
        x = self.col_attn(x, mask = mask) + x
        
        return x

# Main Reformer Class
class ReformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_heads,
        dim_head,
        attn_dropout,
        ff_dropout,
        global_column_attn = False,
        depth = None
    ):
        super().__init__()
        self.layer = nn.ModuleList([
            TriangleMultiplicativeBlock(dim = dim),
            FeedForward(dim = dim, dropout = ff_dropout),
            SeqAttentionBlock(dim = dim, num_heads = num_heads, dim_head = dim_head, dropout = attn_dropout, depth = depth),
            FeedForward(dim = dim, dropout = ff_dropout),
        ])
        
    def forward(self, inputs):
        x, s, mask, seq_mask = inputs
        triangle, ff, seq_attn, seq_ff = self.layer
        
        s = seq_attn(s, mask = seq_mask, pairwise_repr = x)
        s = seq_ff(s) + s
        
        x = triangle(x, mask = mask, s = s, seq_mask = seq_mask)
        x = ff(x) + x
        
        return x, s, mask, seq_mask

class Reformer(nn.Module):
    def __init__(
        self,
        *,
        depth,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([ReformerBlock(**kwargs, depth = d) for d in range(depth)])
        
    def forward(
        self,
        x,
        s,
        mask = None,
        seq_mask = None
    ):
        
        for layer in self.layers:
            x, s, mask, seq_mask = layer((x, s, mask, seq_mask))
        
        return x, s

# Task-wise Heads
class DistogramHead(nn.Module):
    def __init__(self, dim,):
        super().__init__()
        self.to_distogram = FeedForward(dim=dim, dropout=0.1)
        
    def forward(self, x, mask):
        # x: shape [b,L,L,dim]
        # mask: shape [b,L,L], boolean
        # return: shape [b,L,L,d], symmetrized
        
        x = self.to_distogram(x)
        x = x * rearrange(mask, 'b i j -> b i j ()')
        
        return symmetrize_tensor(x)

class FoldingStabilityHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_stability_mean = FeedForward(dim=dim, dropout=0.1)
        
    def forward(self, s, seq_mask):
        # s: shape [b,2,L,dim]
        # seq_mask: shape [b,2,L], boolean
        # return: shape [b,1], shape [b,1,L]
        x_mean = self.to_stability_mean(s)
        
        x_mean = x_mean * rearrange(seq_mask, 'b n i -> b n i ()', n=2)
        
        per_res_stability_mean = reduce(x_mean, 'b n i d -> b 1 i', 'sum', n=2)
        
        folding_stability_mean = reduce(per_res_stability_mean, 'b 1 i -> b 1', 'sum')
        
        return folding_stability_mean, per_res_stability_mean

class FoldingStabilityClassHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_stability_class = FeedForward(dim=dim, dropout=0.1, out_dim=3)
        
    def forward(self, s, seq_mask):
        # s: shape [b,2,L,dim]
        # seq_mask: shape [b,2,L], boolean
        # return: shape [b,3]
        s = self.to_stability_class(s) # [b,2,L,3]
        
        s = s * rearrange(seq_mask, 'b n i -> b n i ()', n=2) # [b,2,L,3]
        
        # [b,2,L,3] -> [b,3]
        folding_stability_class = reduce(s, 'b n i d -> b d', 'sum', n=2)
        
        return folding_stability_class

class SAPHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_sap_mean = FeedForward(dim=dim, dropout=0.1)
        
    def forward(self, s, seq_mask):
        # s: shape [b,2,L,dim]
        # seq_mask: shape [b,2,L], boolean
        # return: shape [b,1], shape [b,1,L]
        x_mean = self.to_sap_mean(s)
        
        x_mean = x_mean * rearrange(seq_mask, 'b n i -> b n i ()', n=2)
        
        per_resi_sap_mean = reduce(x_mean, 'b n i d -> b 1 i', 'sum', n=2)
        
        # rosetta style sap is positive
        return per_resi_sap_mean

class SequenceHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.to_sequence = FeedForward(dim=dim, dropout=0.1, out_dim=20)
        
    def forward(self, x, seq_mask):
        # x: shape [b,2,L,dim]
        # seq_mask: shape [b,2,L], boolean
        # return: shape [b,20,L]
        x = self.to_sequence(x)
        x = x * rearrange(seq_mask, 'b n i -> b n i ()', n=2)
        x = reduce(x, 'b n i d -> b d i', 'sum', n=2)
        
        return x

# Main Model
class IEFFEUM(nn.Module):
    def __init__(
        self, dim=21, depth=5, num_heads=8, dim_head=32, attn_dropout=0, ff_dropout=0):
        super().__init__()
        self.reshape_f_seq = DiffAttn3d(in_dim=1024, out_dim=dim, depth=1)
        self.reshape_f_bb = DiffAttn3d(in_dim=512, out_dim=dim, depth=1)
        self.main_module = Reformer(dim=dim, depth=depth, num_heads=num_heads, dim_head=dim_head, attn_dropout=attn_dropout, ff_dropout=ff_dropout)
        self.distogram_head = DistogramHead(dim)
        self.folding_stability_head = FoldingStabilityHead(dim)
        self.folding_stability_class_head = FoldingStabilityClassHead(dim)
        self.sequence_head = SequenceHead(dim)
        self.sap_head = SAPHead(dim)
        
    def forward(self, x:Tensor, s_1d:Tensor, s_2d:Tensor, mask_1d:Tensor, mask_2d:Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            x: ground truth pair distances, [b,L,L,num_bins]
            s_1d: sequence embedding; prott5, [b,1,L,1024]
            s_2d: backbone embedding; esm-if1, [b,1,L,512]
            mask_1d: 1d boolean masks, [b,2,L]
                1 1 1 0
                1 1 1 0
            mask_2d: 2d boolean masks, [b,L,L]
                1 1 1 0 (padded len = 1)
                1 1 1 0
                1 1 1 0
                0 0 0 0
        Returns:
            distogram: predicted pair distogram, [b,L,L,num_bins]
            (folding_stability, per_res_stability): predicted stability and residue wise contributions
            folding_stability_class: predicted stability class
            per_resi_sap: predicted residue-wise SAP
            seq: reconstructed sequence probability, [b,20,L]
        """
        s_1d = self.reshape_f_seq(s_1d, mask_2d)
        s_2d = self.reshape_f_bb(s_2d, mask_2d)
        s = torch.cat([s_1d, s_2d], dim=1) # [b,2,L,d]
        x = x * rearrange(mask_2d, 'b i j -> b i j ()')
        mask_1d = repeat(mask_1d, 'b 1 i -> b n i', n=2)
        s = s * rearrange(mask_1d, 'b n i -> b n i ()', n=2)
        
        x, s = self.main_module(x, s, mask_2d, mask_1d)
        
        distogram = self.distogram_head(x, mask_2d)
        
        distogram_diagonal = torch.zeros_like(distogram, device=distogram.device)
        diagonal_indices = torch.arange(distogram.shape[1], device=distogram.device)
        distogram_diagonal[:, diagonal_indices, diagonal_indices, 1:] = -1e6
        
        distogram = distogram + distogram_diagonal
        
        folding_stability, per_res_stability = self.folding_stability_head(s, mask_1d)
        folding_stability_class = self.folding_stability_class_head(s, mask_1d)
        
        per_resi_sap = self.sap_head(s, mask_1d)
        seq = self.sequence_head(s, mask_1d)
        
        assert not torch.isnan(folding_stability[0]).any()
        return distogram, (folding_stability, per_res_stability), folding_stability_class, per_resi_sap, seq