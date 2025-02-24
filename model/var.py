import torch, math
import torch.nn as nn
from functools import partial
from typing import Tuple

from model.var_basics import AdaLNSelfAttn, AdaLNBeforeHead, SharedAdaLin
from model.var_helpers import sample_with_top_k_top_p


class VAR(nn.Module):
    def __init__(self,
        vqvgae, config, 
        drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0, cond_drop_rate=0.1,
        attn_l2_norm=False, norm_eps=1e-6,
    ):
        super().__init__()
        
        # Hyperparameters
        self.depth = config.var.depth
        self.num_heads = config.var.num_heads
        self.C = self.D = config.var.emb_dim

        self.Cvae = vqvgae.quantizer.embedding_dim
        self.V = vqvgae.quantizer.codebook_size

        self.scales = config.vqvgae.quantizer.scales
        self.L = sum(self.scales)                
        self.first_l = self.scales[0]   # Size of the first scale (will be replaced by a class label of the same size during training)
        self.num_scales_minus_1 = len(self.scales) - 1

        self.num_classes = config.var.num_classes # For conditional generation
        self.cond_drop_rate = cond_drop_rate

        # Input (word) embedding
        self.word_embed = nn.Linear(self.Cvae, self.C)
        quant: VectorQuantizer = vqvgae.quantizer
        self.vae_quant_proxy: Tuple[VectorQuantizer] = (quant, )
        self.vae_proxy: Tuple[VQVGAE] = (vqvgae, )
        
        # Class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.class_embed = nn.Embedding(self.num_classes + 1, self.C) # +1 for no class label
        nn.init.trunc_normal_(self.class_embed.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # Absolute position embedding
        self.pos_1LC = nn.Parameter(torch.empty(1, self.L, self.C))
        nn.init.trunc_normal_(self.pos_1LC.data, mean=0, std=init_std)
        self.lvl_embed = nn.Embedding(len(self.scales), self.C) # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)

        # Backbone
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C))
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)

        # Backbone
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, 
                num_heads=self.num_heads, mlp_ratio=config.var.train.mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path=dpr[block_idx], last_drop_p=0 if block_idx==0 else dpr[block_idx-1], attn_l2_norm=attn_l2_norm,
            )
            for block_idx in range(self.depth)
        ])
        
        # Attention mask used in training
        d = torch.cat([torch.full((pn,), i) for i, pn in enumerate(self.scales)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())

        # Classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)
    

    def forward(self, x_BLCv_wo_first_l, label_B):
        B = x_BLCv_wo_first_l.shape[0]
        
        with torch.cuda.amp.autocast(enabled=False):
            
            # SOS token
            label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, 0, label_B) # B, 1
            
            sos = cond_BD = self.class_embed(label_B) # B, C
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1) # B, 1, C => SOS embedding (not token anymore)

            # Whole sequence (SOS + sequence without first l)
            x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1) # B, 1+L-1 (sos+l_wo_frst), C
            x_BLC += self.lvl_embed(self.lvl_1L.expand(B, -1)) + self.pos_1LC # Add positional embedding => B, L, C

        # Masking and SharedAdaLN
        attn_bias = self.attn_bias_for_masking # 1, 1, L, L
        cond_BD_or_gss = self.shared_ada_lin(cond_BD) # B, 1, 6, C

        # Hack: get the dtype if mixed precision is used (TODO: might be able to remove this)
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        # Backbone
        for block in self.blocks:
            x_BLC = block(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        x_BLC = self.head(self.head_nm(x_BLC.float(), cond_BD).float()).float() # codeword for each L => B, L, V
        
        return x_BLC

    def autoregressive_infer_cfg(self, B, label_B, cfg, top_k, top_p):
        sos = cond_BD = self.class_embed(torch.cat((label_B, torch.full_like(label_B, fill_value=0)), dim=0))

        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2*B, self.first_l, -1) + self.pos_start.expand(2*B, self.first_l, -1) + lvl_pos[:, :self.first_l]

        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.scales[-1])
        for block in self.blocks:
            block.attn.kv_caching(True)
        
        for si, pn in enumerate(self.scales):
            ratio = si / self.num_scales_minus_1
            cur_L += pn

            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            for block in self.blocks:
                x = block(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            logits_BLV = self.head(self.head_nm(x.float(), cond_BD).float()).float() # codeword for each L => B, L, V

            t = cfg * ratio
            logits_BLV = (1+t) * logits_BLV[:B] - t * logits_BLV[B:]

            idx_Bl = sample_with_top_k_top_p(logits_BLV, rng=None, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            h_BCn = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            
            h_BCn = h_BCn.transpose_(1, 2).reshape(B, self.Cvae, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.scales), f_hat, h_BCn)
            if si != self.num_scales_minus_1:
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.scales[si+1]]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        
        for block in self.blocks:
            block.attn.kv_caching(False)
        
        return self.vae_proxy[0].fhat_to_graph(f_hat, original_sizes=label_B)
