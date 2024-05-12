import torch
import torch.cuda.nvtx as nvtx
import networkx as nx
import itertools, time
import torch.nn as nn
import copy
from einops import rearrange
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional
import torchvision.models as models
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)

def get_fpn_backbone():
    resnet18 = models.resnet18(pretrained=True)
    backbone = resnet_fpn_backbone('resnet18', weights = resnet18, pretrained=True, \
                                   trainable_layers=3, returned_layers=[1])
    # import torchvision
    # backbone = torchvision.ops.FeaturePyramidNetwork([10, 20, 30], 5)
    return backbone

class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[Tensor] = None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        return weights

class Norm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)
def find(parent, i):
    if parent[i]==i:
        return i
    return find(parent, parent[i])

def union(parent, rank, x, y):
    xroot = find(parent,x)
    yroot = find(parent, y)

    if rank[xroot] < rank[yroot]:
        parent[xroot]=yroot
    elif  rank[xroot] > rank[yroot]:
        parent[yroot]=xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1
    return parent, rank

def merge_pairs_check(pairs):
    G = nx.Graph()
    for pair in pairs:
        G.add_edge(*pair)

    merged_groups = list(nx.connected_components(G))
    del G
    return merged_groups

def merge_pairs(merge_elements):

    merge_elements = [(int(x[0]), int(x[1])) for x in merge_elements]
    return merge_pairs_check(merge_elements)

def is_valid_input(added_tokens, input_tuple):
    return not(input_tuple in added_tokens or input_tuple[::-1] in added_tokens)

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class SelfAttentionMasked(nn.Module):
    def __init__(self, dim, num_heads=2, dim_head_output=64, dropout=0.):
        super().__init__()
        
        self.num_heads = num_heads
        # \sqrt{d_{k}}
        self.att_scale = dim_head_output ** (-0.5) 
        self.attention_fn = nn.Softmax(dim=-1)
        self.linear_projection_kqv = nn.Linear(dim, num_heads * dim_head_output * 3, bias=False)

        # We need to combine the output from all heads
        self.output_layer = nn.Sequential(nn.Linear(num_heads * dim_head_output, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        # qkv should be (..., seq_len, num_heads * dim_head_output)
        qkv = self.linear_projection_kqv(x).chunk(3, dim=-1)

        # We need to convert to (..., num_heads, seq_len, dim_head_output)
        # By doing this operation, we assume q, k, v have the same dimension. But this is not necessarily the case
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        
        # q.dot(k.transpose)
        qk_dot_product = torch.matmul(q, k.transpose(-1, -2)) * self.att_scale
        if mask is not None and len(mask.shape)==3:
            tmp_mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            qk_dot_product = qk_dot_product.masked_fill(tmp_mask==1., -1e10)
        self.att_weights = self.attention_fn(qk_dot_product)
        # (..., num_heads, seq_len, dim_head_output)
        out = torch.matmul(self.att_weights, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        # Merge the output from heads to get single vector
        return self.output_layer(out)

class LinearProjection(nn.Module):
    def __init__(self,
                 input_shape,
                 out_dim,
                 bias=True):
        super().__init__()

        in_dim = input_shape[-1]
        self.out_dim = out_dim
        self.projection = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        out = self.projection(x)
        return out

class TransformerFeedForwardNN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        # Remember the residual connection
        self.layers = [nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout)]
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.net(x)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
class TransformerEncoder_gos_and_masking(nn.Module):
    def __init__(self,
                 input_dim,
                 num_layers,
                 num_heads,
                 dim_head_output,
                 mlp_dim,
                 dropout,
                 prune_num,
                 keep_num,
                 **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.dim_head_output = dim_head_output
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList([])
        self.drop_path = DropPath(dropout) if dropout > 0. else nn.Identity()
        self.prune_num = [prune_num.get(str(i), 0) for i in range(num_layers)]
        self.keep_num = [keep_num.get(str(i), 0) for i in range(num_layers)]
        print(f"PRUNING NUM is {self.prune_num}")
        print(f"KEEPING NUM is {self.keep_num}")
        self.attention_output = {}
        for idx in range(num_layers):
            if self.keep_num[idx] != 0:
                self.layers.append(nn.ModuleList([
                    Norm(input_dim),
                    SelfAttention_with_gos_and_masking(input_dim, num_heads=num_heads, dim_head_output=dim_head_output, dropout=dropout,\
                                               prune_num = self.prune_num[idx], keep_num = self.keep_num[idx]),
                    Norm(input_dim),
                    TransformerFeedForwardNN(input_dim, mlp_dim, dropout=dropout)
                ]))
            else:
                self.layers.append(nn.ModuleList([
                    Norm(input_dim),
                    SelfAttentionMasked(input_dim, num_heads=num_heads, dim_head_output=dim_head_output, dropout=dropout),
                    Norm(input_dim),
                    TransformerFeedForwardNN(input_dim, mlp_dim, dropout=dropout)
                ]))
            
        self.fuse_mapping_post = LinearProjection(input_shape=(input_dim,), out_dim=input_dim)
    def forward(self, x, mask=None, abstract_info = None):
        self.num_img, self.num_mask, _ = x.shape
        self.num_mask -= 1
        # self.survived_mask_idxs ={}
        # self.survived_mask_idxs['-1'] = [[i for i in range(self.num_mask+1)] for _ in range(self.num_img)]
        for layer_idx, (att_norm, att, ff_norm, ff) in enumerate(self.layers):
            nvtx.range_push(f'Transformer layer {layer_idx}')
            if isinstance(att, SelfAttention_with_gos_and_masking):
                nvtx.range_push(f'1. check')
                tmp, what_to_prune, what_to_merge, how_to_merge, new_mask = att(att_norm(x), mask)
                self.now_keep_num = self.keep_num[layer_idx]
                nvtx.range_pop()
                x = x + drop_path(tmp)
                nvtx.range_push(f'2. merge tokens')
                # self.check_left_tokens(prune= self.prune_num[layer_idx] == 0,what_to_prune=what_to_prune, what_to_merge = what_to_merge)
                x = self.merge_tokens(x, what_to_prune, what_to_merge, how_to_merge, mask, self.now_keep_num, abstract_info, layer_idx)
                # self.survived_mask_idxs[str(layer_idx)] = survived_mask_idx
                nvtx.range_pop()
                x = x + self.drop_path(ff(ff_norm(x)))
                mask = new_mask
            elif isinstance(att, SelfAttentionMasked):
                tmp = att(att_norm(x), mask)
                x = x + drop_path(tmp)
                x = x + self.drop_path(ff(ff_norm(x)))
            nvtx.range_pop()
            # print(x.shape)
        return x# , self.survived_mask_idxs[str(layer_idx)]
    
    def check_left_tokens(self, prune, what_to_prune, what_to_merge):
        for img_idx in range(self.num_img):
            idx_pruned = what_to_prune[img_idx]
            if prune == True:
                for i in range(len(self.survived_mask_idxs[img_idx])):
                    if i not in idx_pruned:
                        breakpoint()
                    else:   
                        breakpoint()
            else:
                self.survived_mask_idxs[img_idx] = sorted([self.survived_mask_idxs[img_idx][i-1] for i in range(len(self.survived_mask_idxs[img_idx])) if i not in idx_pruned])
        return 
    
    def merge_tokens(self, batch_data, what_to_prune, what_to_merge, how_to_merge, mask, keep_num, abstract_info, layer_idx):
        """
        Merge k most similar token pairs in each batch based on cosine similarity.

        Parameters:
        - batch_data: Tensor of shape (B, N, D) where B is the batch size, N is the number of tokens, and D is the dimension of each token.
        - k: Number of token pairs to merge
        
        Returns:
        - Merged data for each batch
        """
        B, N, _ = batch_data.shape
        # previous_layer_idx = self.survived_mask_idxs[str(layer_idx-1)]
        if self.prune_num[layer_idx] >0:
            # print("=================== Pruning ========================")
            # survived_batch = []
            remaining_tokens_mask = torch.ones((B,N), dtype=bool).to(batch_data.device)
            for batch_idx, prune_elems in enumerate(what_to_prune):
                remaining_tokens_mask[batch_idx, prune_elems]=False
                # current_layer_mask_idx = copy.deepcopy(previous_layer_idx[batch_idx])
                # print(f'Batch idx {batch_idx} with {prune_elems}')
                # print(f'\tPrevious batch idx: {current_layer_mask_idx}')
                # try:
                #     current_layer_mask_idx = [item for idx, item in enumerate(current_layer_mask_idx) if idx not in prune_elems]
                # except IndexError:
                #     breakpoint()
                # print(f'\tModified batch idx: {current_layer_mask_idx}')
                # survived_batch.append(current_layer_mask_idx)
            merged_batches = [batch_data[batch_idx][mask].to(batch_data.device) for batch_idx, mask in enumerate(remaining_tokens_mask)]
        else:
            # print("=================== Merging ========================")
            merged_batches = []
            # survived_batch = []
            for batch_idx in range(B):
                # k = int(what_to_merge.shape[1])
                token_vectors = batch_data[batch_idx]
                merge_elements = what_to_merge[batch_idx]
                prune_elements = what_to_prune[batch_idx]
                indices = N-1 #not considering goal token
                merge_elements = [(int(x[0]), int(x[1])) for x in merge_elements]
                merge_elements = merge_pairs_check(merge_elements) 
                merge_elements_tmp = list(itertools.chain(*merge_elements))
                
                merge_weights = how_to_merge[batch_idx]
                # survived_each_batch_idx = set()
                new_vectors = []
                remaining_tokens_mask = torch.ones(N, dtype=bool).to(batch_data.device)
                remaining_tokens_mask[merge_elements_tmp]= False # action token id still true
                remaining_tokens_mask[indices+1:]=False
                remaining_vectors_tensor = token_vectors[remaining_tokens_mask]
                for merge_unit in merge_elements:
                    new_vector=[]
                    attn_weight=[]
                    final_vector=[]
                    # add_survived_mask = []
                    for i in list(merge_unit):
                        x,y = torch.argwhere(what_to_merge[batch_idx]==i)[0].tolist()
                        attn_weight.append(merge_weights[x][y])
                        new_vector.append(token_vectors[i])
                    attn_weight = torch.softmax(torch.tensor(attn_weight), dim=-1).to(batch_data.device)
                    for score, vector in zip(attn_weight, new_vector):
                        final_vector.append(score*vector)
                    new_vector = self.fuse_mapping_post(torch.sum(torch.stack(final_vector, dim=0),dim=0).to(batch_data.device))
                    new_vectors.append(new_vector)
                    # print(merge_unit, previous_layer_idx[batch_idx])
                #     for idx in merge_unit:
                #         breakpoint()
                #         if previous_layer_idx[batch_idx][idx-1] is not list:
                #             add_survived_mask.add(previous_layer_idx[batch_idx][idx-1])
                #         else:
                #             add_survived_mask.update(set(previous_layer_idx[batch_idx][idx-1]))
                #     survived_each_batch_idx.append(add_survived_mask)
                # survived_each_batch_idx.update(set([idx+1 for idx, i in enumerate(remaining_tokens_mask[1:]) if i==True]))
                # survived_batch.append(survived_each_batch_idx)
                # print(f'Batch idx {batch_idx} with {merge_unit}')
                # print(f'{survived_each_batch_idx}')
                current_batch_vector = torch.cat((torch.stack(new_vectors), remaining_vectors_tensor), dim=0)
                del new_vectors, remaining_tokens_mask,merge_elements_tmp,attn_weight
                merged_batches.append(current_batch_vector)
            del token_vectors, merge_elements, prune_elements, current_batch_vector
        merged_batches = torch.stack(merged_batches)
        return merged_batches# , survived_batch
    
class SelfAttention_with_gos_and_masking(nn.Module):
    def __init__(self, dim, num_heads=8, dim_head_output=64, dropout=0., prune_num=0.5, keep_num=3):
        super().__init__()
        
        self.num_heads = num_heads
        # \sqrt{d_{k}}
        self.att_scale = dim_head_output ** (-0.5) 
        self.attention_fn = nn.Softmax(dim=-1)
        self.linear_projection_kqv = nn.Linear(dim, num_heads * dim_head_output * 3, bias=False)
        self.prune_num = prune_num
        self.keep_num = keep_num
        # We need to combine the output from all heads
        self.output_layer = nn.Sequential(nn.Linear(num_heads * dim_head_output, dim), nn.Dropout(dropout))


    def forward(self, x, mask):
        # qkv should be (..., seq_len, num_heads * dim_head_output)
        qkv = self.linear_projection_kqv(x).chunk(3, dim=-1)

        # We need to convert to (..., num_heads, seq_len, dim_head_output)
        # By doing this operation, we assume q, k, v have the same dimension. But this is not necessarily the case
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        qk_dot_product = torch.matmul(q, k.transpose(-1, -2)) * self.att_scale
        if mask is not None and len(mask.shape)==3:
            tmp_mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            qk_dot_product = qk_dot_product.masked_fill(tmp_mask==1., -1e10)
            # qk_dot_product += (tmp_mask==1) * -1e10
        self.att_weights = self.attention_fn(qk_dot_product)
        out = torch.matmul(self.att_weights, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.output_layer(out)
        # MERGE BASED 1) attention weights
        # out, mask = self.merge_tokens_based_on_topk_attention(out, mask, k = 2)#int(N*(1-self.keep_rate)))
        # MERGE BASED 2) similarity metric
        self.att_weights_topk= self.att_weights.mean(dim=1)[:,0] # Get the action-token attention weights.
        self.att_weights_topk[:, 0] = float('-inf')
        what_to_prune, what_to_merge, how_to_merge, masks = self.determine_fuse(out, mask)
        return out, what_to_prune, what_to_merge, how_to_merge, masks
    
    def determine_fuse(self, batch_data, mask):
        """
        Merge k most similar token pairs in each batch based on cosine similarity.

        Parameters:
        - batch_data: Tensor of shape (B, N, D) where B is the batch size, N is the number of tokens, and D is the dimension of each token.
        - k: Number of token pairs to merge
        
        Returns:
        - Merged data for each batch
        """
        B, N, _ = batch_data.shape
        # Normalize each token tensor to use dot product for cosine similarity
        normalized_data = F.normalize(batch_data, p=2, dim=2)
        # Calculate cosine similarity
        cosine_sim = torch.bmm(normalized_data, normalized_data.transpose(1, 2))  # (B, N, N)

        nvtx.range_push(f'topk')
        # Fill diagonal with large negative value to avoid selecting same tokens
        indices = torch.arange(N).to(batch_data.device)
        cosine_sim[:, indices, indices] = float('-inf')
        cosine_sim[:, 0] = float('-inf')
        cosine_sim[:, :, 0] = float('-inf')
        mask = torch.zeros(cosine_sim.shape).to('cuda')
        cosine_sim = cosine_sim.masked_fill(mask==1., float('-inf'))
        # Get top k similar pairs 
        _, attentive_top_k_indices = torch.topk(self.att_weights_topk, N, dim=1)
        _, top_k_indices = torch.topk(cosine_sim.view(B, -1), (N-1)**2, dim=1)
        
        nvtx.range_pop()
        what_to_prune = []
        what_to_merge = []
        how_to_merge = []
        indices = batch_data.shape[1] -1 #not considering goal token

        remove_num = self.prune_num
        merge_num = indices - self.keep_num - remove_num
        survived_mask = torch.full((B, 1+self.keep_num, 1+self.keep_num), 0).to(batch_data.device)
        if remove_num > 0:
            # Only pruning
            for batch_idx in range(B):
                pruned_tokens = attentive_top_k_indices[batch_idx][:indices][-remove_num:]
                what_to_prune.append(pruned_tokens.to(batch_data.device))
                what_to_merge.append(torch.tensor([-1]).to(batch_data.device))
                how_to_merge.append(torch.tensor([-1]).to(batch_data.device))
            del pruned_tokens
            del attentive_top_k_indices, top_k_indices
            return what_to_prune, what_to_merge, how_to_merge, survived_mask
        else:        
            # Only merging
            for batch_idx in range(B):
                # no pruning
                pruned_tokens = torch.tensor([-1]).to(batch_data.device)
                what_to_prune.append(torch.tensor(pruned_tokens).to(batch_data.device))

                # determine what to merge
                merged_tokens = []
                merged_tokens_attention = []
                attentive_tokens = attentive_top_k_indices[batch_idx][:indices][:indices-remove_num]
                for index in top_k_indices[batch_idx]:
                    i, j = index.item() // N, index.item() % N
                    if (i in attentive_tokens) and (j in attentive_tokens)and i!=j and (i != 0) and (j!=0):
                        if is_valid_input(merged_tokens, (i,j)):
                            merged_tokens = merged_tokens + [(i, j)] #index 
                            merged_tokens_attention = merged_tokens_attention + \
                                [(torch.tensor(self.att_weights_topk[batch_idx,i]), torch.tensor(self.att_weights_topk[batch_idx,j]))]
                            if len(merged_tokens) >= merge_num:
                                tmp = [(int(x[0]), int(x[1])) for x in merged_tokens]
                                tmp_left = merge_pairs_check(tmp) 
                                tmp_entire = list(itertools.chain(*tmp_left))
                                if len(tmp_entire) - len(tmp_left) == merge_num:
                                    break
                what_to_merge.append(torch.tensor(merged_tokens).to(batch_data.device))
                how_to_merge.append(merged_tokens_attention)
            del pruned_tokens, merged_tokens, merged_tokens_attention
            del attentive_top_k_indices, top_k_indices
            return what_to_prune, what_to_merge, how_to_merge, survived_mask

