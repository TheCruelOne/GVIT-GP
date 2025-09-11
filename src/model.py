import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.): super().__init__(); self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training: return x
        keep_prob = 1 - self.drop_prob;
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device);
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__();
        out_features = out_features or in_features;
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features);
        self.act = act_layer();
        self.fc2 = nn.Linear(hidden_features, out_features);
        self.drop = nn.Dropout(drop)
    def forward(self, x): x = self.fc1(x); x = self.act(x); x = self.drop(x); x = self.fc2(x); x = self.drop(x); return x

class RelativeSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., max_relative_position=64):
        super().__init__();
        self.num_heads = num_heads;
        head_dim = dim // num_heads;
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias);
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim);
        self.proj_drop = nn.Dropout(proj_drop)
        self.max_relative_position = max_relative_position
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * max_relative_position + 1), num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
    def forward(self, x, return_attention=False):
        B, N, C = x.shape;
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        q_indices = torch.arange(N, device=x.device).view(-1, 1);
        k_indices = torch.arange(N, device=x.device).view(1, -1)
        relative_indices = k_indices - q_indices;
        relative_indices = relative_indices + self.max_relative_position
        relative_indices = torch.clamp(relative_indices, 0, 2 * self.max_relative_position)
        pos_bias = self.relative_position_bias_table[relative_indices];
        pos_bias = pos_bias.permute(2, 0, 1).unsqueeze(0)
        attn = attn + pos_bias
        attn = attn.softmax(dim=-1);
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C);
        x = self.proj(x);
        x = self.proj_drop(x)
        if return_attention: return x, attn
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, max_relative_position=64):
        super().__init__();
        self.norm1 = norm_layer(dim)
        self.attn = RelativeSelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                          proj_drop=drop, max_relative_position=max_relative_position)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim);
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
    def forward(self, x, return_attention=False):
        if return_attention:
            attn_output, attn_weights = self.attn(self.norm1(x), return_attention=True)
            x = x + self.drop_path(attn_output);
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn_weights
        x = x + self.drop_path(self.attn(self.norm1(x)));
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__();
        self.num_heads = num_heads;
        head_dim = dim // num_heads;
        self.scale = head_dim ** -0.5
        self.query = nn.Linear(dim, dim, bias=qkv_bias);
        self.key = nn.Linear(dim, dim, bias=qkv_bias)
        self.value = nn.Linear(dim, dim, bias=qkv_bias);
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim);
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x_query, x_kv):
        B, N_q, C = x_query.shape;
        _, N_kv, _ = x_kv.shape
        q = self.query(x_query).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.key(x_kv).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.value(x_kv).reshape(B, N_kv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale;
        attn = attn.softmax(dim=-1);
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C);
        x = self.proj(x);
        x = self.proj_drop(x)
        return x

class MLP_grm(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, drop_ratio=0.1):
        super().__init__();
        layers = [];
        in_dim = input_dim
        for h_dim in hidden_layers: layers.append(nn.Linear(in_dim, h_dim)); layers.append(nn.ReLU()); layers.append(
            nn.Dropout(drop_ratio)); in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim));
        self.mlp = nn.Sequential(*layers)
    def forward(self, x): return self.mlp(x)

class ConvPatchEmbed(nn.Module):
    def __init__(self, num_snp_values, embed_dim, patch_size, padding_idx):
        super().__init__()
        self.patch_size = patch_size;
        self.padding_idx = padding_idx
        self.internal_embed = nn.Embedding(num_snp_values, embed_dim, padding_idx=self.padding_idx)
        self.proj = nn.Conv1d(embed_dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        B, S = x.shape
        padding_size = (self.patch_size - S % self.patch_size) % self.patch_size
        if padding_size > 0: x = F.pad(x, (0, padding_size), "constant", self.padding_idx)
        x = self.internal_embed(x);
        x = x.permute(0, 2, 1)
        x = self.proj(x);
        x = x.permute(0, 2, 1);
        x = self.norm(x)
        return x

class CrossAttentionFusionModel(nn.Module):
    def __init__(self, num_snp_values, grm_vector_len, num_outputs, embed_dim, depth, num_heads, mlp_ratio,
                 drop_ratio, attn_drop_ratio, fusion_start_index, grm_mlp_hidden_layers, max_relative_position,
                 patch_size, padding_idx):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = ConvPatchEmbed(num_snp_values=num_snp_values, embed_dim=embed_dim, patch_size=patch_size,
                                          padding_idx=padding_idx)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)
        dpr = [x.item() for x in torch.linspace(0, drop_ratio, depth)]
        block_creator = partial(Block, dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=True,
                                drop=drop_ratio, attn_drop=attn_drop_ratio, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                max_relative_position=max_relative_position)
        self.blocks_pre_fusion = nn.ModuleList([block_creator(drop_path=dpr[i]) for i in range(fusion_start_index)])
        self.mlp_grm = MLP_grm(input_dim=grm_vector_len, hidden_layers=grm_mlp_hidden_layers, output_dim=embed_dim)
        self.fusion_norm1 = nn.LayerNorm(embed_dim);
        self.fusion_norm2 = nn.LayerNorm(embed_dim)
        self.cross_attention = CrossAttention(dim=embed_dim, num_heads=num_heads, attn_drop=attn_drop_ratio,
                                              proj_drop=drop_ratio)
        self.fusion_drop_path = DropPath(dpr[fusion_start_index]) if fusion_start_index < depth else nn.Identity()
        self.blocks_post_fusion = nn.ModuleList(
            [block_creator(drop_path=dpr[i]) for i in range(fusion_start_index, depth)])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.GELU(), nn.Dropout(0.2),
                                  nn.Linear(embed_dim // 2, num_outputs))
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0);
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Embedding, nn.Conv1d)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'padding_idx') and m.padding_idx is not None:
                with torch.no_grad(): m.weight[m.padding_idx].fill_(0)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, snp_input, grm_vector, return_attention=False):
        B = snp_input.shape[0];
        x = self.patch_embed(snp_input)
        cls_tokens = self.cls_token.expand(B, -1, -1);
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)
        for blk in self.blocks_pre_fusion: x = blk(x)
        h_vit = x
        h_grm = self.mlp_grm(grm_vector).unsqueeze(1)
        fused_h = h_vit + self.fusion_drop_path(
            self.cross_attention(self.fusion_norm1(h_vit), self.fusion_norm2(h_grm)))
        final_output = fused_h
        for i, blk in enumerate(self.blocks_post_fusion):
            if i == len(self.blocks_post_fusion) - 1 and return_attention:
                final_output, attn_weights = blk(final_output, return_attention=True)
            else:
                final_output = blk(final_output)
        final_output = self.final_norm(final_output)
        cls_output = final_output[:, 0]
        prediction = self.head(cls_output)
        if return_attention: return prediction, attn_weights
        return prediction