# --------------------------------------------------------
# Based on MAE, CPC, BEiT, timm, DINO and DeiT code bases
# https://github.com/pengzhiliang/MAE-pytorch
# https://github.com/jefflai108/Contrastive-Predictive-Coding-PyTorch
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
import scipy.stats as stats
from modeling_finetune import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from utils import generate_gaussian_kernel


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


__all__ = [
    'pretrain_mae_base_patch16_224', 
    'pretrain_mae_large_patch16_224', 
]

class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=11, patch_size=11, embed_dim=200, num_classes=0, in_chans=121, depth=12, sigma=1,
                 num_heads=12, use_model='neicot_pre', mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_learnable_pos_emb=False):
        super().__init__()       

        k = int((patch_size - 1)/2)    # 卷积核为3，步长为1
        self.timestep = int(k**2)
        print("timestep的取值",self.timestep)

        self.use_model = use_model
        self.num_classes = num_classes
        self.num_features = self.in_chans = in_chans  # num_features for consistency with other models
        
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, embed_dim=embed_dim, in_chans=in_chans)
        dim_control = self.patch_embed.dim_control

        sigma = sigma
        kernel = generate_gaussian_kernel(k, sigma)
        self.val_gauss = kernel*(1/kernel.max())
        num_patches = self.patch_embed.num_patches

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim_control))
        else:
            self.pos_embed = get_sinusoid_encoding_table(num_patches, dim_control)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=dim_control, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.conv_norm = norm_layer(dim_control)
        self.norm =  norm_layer(dim_control)

        if use_model == 'neicot_pre':
            self.Wk  = nn.ModuleList([nn.Linear(dim_control, dim_control) for i in range(self.timestep)])
            self.softmax  = nn.Softmax()
            self.lsoftmax = nn.LogSoftmax()

        self.head = nn.Linear(dim_control, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)  # 将网络结构给打出来

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.in_chans, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask):
        x, x_en = self.patch_embed(x)
        B, _, C = x.shape
        x = self.conv_norm(x)
        if self.use_model == 'neicot_pre':
            noice_nce = self.val_gauss
            nce = 0
            encode_samples = torch.empty((self.timestep, B, C)).float()     # e.g. 61*64*200
            for i in np.arange(0, self.timestep):
                encode_samples[i] = x_en[:,i,:].view(B, C)
            
            pred = torch.empty((self.timestep, B, C)).float()
            anchor = x.mean(dim = 1)
            for i in np.arange(0, self.timestep):
                linear = self.Wk[i]
                pred[i] = linear(anchor)
            for i in np.arange(0, self.timestep):
                total = torch.mm(encode_samples[i], torch.transpose(pred[i],0,1))   # e.g. size 64*64
                correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, B))) # correct is a tensor
                nce += torch.sum(torch.diag(self.lsoftmax(total)))*noice_nce[i]     # nce is a tensor + gauss_weight
            nce /= -1.*B*self.timestep
            acc = 1.*correct.item()/B

        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible
        for blk in self.blocks:
            x_vis = blk(x_vis)
        x_vis = self.norm(x_vis)
        
        if self.use_model == 'neicot_pre':
            return x_vis, nce, acc  
        else:
            return x_vis

    def forward(self, x, mask):
        if self.use_model == 'neicot_pre':
            x, nce, acc = self.forward_features(x, mask)
            x = self.head(x)
            return x, nce, acc
        else: 
            x = self.forward_features(x, mask)
            x = self.head(x)
            return x

    
class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=11, num_classes=200, in_chans=200, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, num_patches=200,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.in_chans = in_chans  # num_features for consistency with other models
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=in_chans, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(in_chans)
        self.head = nn.Linear(in_chans, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.in_chans, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num):
        for blk in self.blocks:
            x = blk(x)
        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:]))           
        else:
            x = self.head(self.norm(x))
        return x

class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=11, 
                 patch_size=11, 
                 encoder_in_chans=200, 
                 encoder_num_classes=0, 
                 encoder_embed_dim=121, 
                 encoder_depth=3,
                 encoder_sigma=1,
                 encoder_num_heads=12, 
                 use_model='neicot_pre',
                 decoder_num_classes=121, 
                 decoder_embed_dim=1000, 
                 decoder_depth=1,
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 num_classes=0, # avoid the error from create_fn in timm
                 embed_dim=0, # avoid the error from create_fn in timm
                 ):
        self.use_model = use_model
        self.patch = patch_size
        super().__init__()      # 调用父类的init方法
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size, 
            patch_size=patch_size, 
            embed_dim=encoder_in_chans, 
            num_classes=encoder_num_classes, 
            in_chans=encoder_embed_dim, 
            depth=encoder_depth,
            sigma=encoder_sigma,
            num_heads=encoder_num_heads, 
            use_model=use_model,
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb)

        self.decoder = PretrainVisionTransformerDecoder(
            patch_size=patch_size, 
            num_patches=self.encoder.patch_embed.num_patches,
            num_classes=decoder_num_classes, 
            in_chans=decoder_embed_dim, 
            depth=decoder_depth,
            num_heads=decoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values)
        
        self.encoder_to_decoder = nn.Linear(self.encoder.patch_embed.dim_control , decoder_embed_dim, bias=False)   
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.pos_embed = get_sinusoid_encoding_table(self.encoder.patch_embed.num_patches, int(decoder_embed_dim))   
        trunc_normal_(self.mask_token, std=.02)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def forward(self, x, mask):
        if self.use_model == 'neicot_pre':
            x_vis, nce, acc = self.encoder(x, mask)
        else:
            x_vis = self.encoder(x, mask)
        x_vis = self.encoder_to_decoder(x_vis)

        B, N, C = x_vis.shape

        expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_emd_vis = expand_pos_embed[~mask].reshape(B, -1, C)
        pos_emd_mask = expand_pos_embed[mask].reshape(B, -1, C)

        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)
        x = self.decoder(x_full, pos_emd_mask.shape[1])
        
        if self.use_model == 'neicot_pre':
            return x, nce, acc
        else:
            return x

# HSI_dataset
@register_model
def pretrain_mae_3_1_patch(pretrained=False, **kwargs):
    # 这里要注意depth和head的数要和finetuing的对的上，不然无法匹配加载。
    encoder_in_chans = kwargs['patch_size']**2
    model = PretrainVisionTransformer(
        img_size=11,
        patch_size=kwargs['patch_size'], 
        encoder_in_chans=encoder_in_chans, 
        encoder_embed_dim=kwargs['n_bands'], 
        encoder_depth=kwargs['depth'], 
        encoder_sigma=kwargs['sigma'],
        encoder_num_heads=6,
        encoder_num_classes=0,
        use_model=kwargs['use_model'],
        decoder_num_classes=kwargs['n_bands'],
        decoder_embed_dim=1000,
        decoder_depth=1,
        decoder_num_heads=6,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    print("depth=",kwargs['depth'])
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


