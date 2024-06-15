# --------------------------------------------------------
# Based on MAE, CPC, BEiT, timm, DINO and DeiT code bases
# https://github.com/pengzhiliang/MAE-pytorch
# https://github.com/jefflai108/Contrastive-Predictive-Coding-PyTorch
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from utils import generate_gaussian_kernel, plot_correlation_matrix


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 17, 'input_size': (200, 11, 11), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=6, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)       # dim 是一个patch内的像素个数，升维操作。
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size=11, embed_dim=121, in_chans=200):
        super().__init__()
        self.dim_control = in_chans
        patch_size = to_2tuple(patch_size)
        num_patches = embed_dim
        self.patch_shape = embed_dim
        self.patch_size = patch_size 
        self.num_patches = num_patches

        self.td_conv = nn.Sequential(
            nn.Conv2d(in_chans, self.dim_control, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.dim_control),   # 归一化
            nn.ReLU(inplace=True),
        )

        self.max_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3,3),stride=2),
            )

    def forward(self, x, **kwargs):
        x = self.td_conv(x)
        x_en = self.max_pool(x).flatten(2).transpose(1,2)
        x = x.flatten(2).transpose(1,2)
        return x, x_en
    
# sin-cos position encoding
def get_sinusoid_encoding_table(position, hid): 
    n_position = position
    d_hid = hid
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 patch_size=11, 
                 embed_dim=121, 
                 num_classes=17, 
                 in_chans=200,
                 depth=3,
                 num_heads=12, 
                 model_name='neicot_liner',
                 sigma=7,
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False, 
                 init_scale=0.,
                 use_mean_pooling=True):
        super().__init__()
        self.model_name = model_name

        # 设置需要预测token的patch大小，根据x_en做的操作控制
        k = int((patch_size - 1)/2)    # 卷积核为3，步长为2
        self.timestep = int(k**2)
        print("timestep的取值:",self.timestep)

        self.num_classes = num_classes
        self.num_features = self.in_chans = in_chans
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, embed_dim=embed_dim, in_chans=in_chans)
        dim_control = self.patch_embed.dim_control              # 添加的控制升维的参数
        num_patches = self.patch_embed.num_patches
        kernel = generate_gaussian_kernel(k, sigma)
        self.val_gauss = kernel*(1/kernel.max())

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, in_chans))
        else:
            self.pos_embed = get_sinusoid_encoding_table(num_patches, dim_control)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.ModuleList([
            Block(
                dim=dim_control, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        
        self.conv_norm = norm_layer(dim_control)
        
        if model_name == 'neicot_liner':
            self.Wk = nn.ModuleList([nn.Linear(dim_control, dim_control) for i in range(self.timestep)])
            self.softmax  = nn.Softmax()
            self.lsoftmax = nn.LogSoftmax()

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(dim_control)
        self.fc_norm = norm_layer(dim_control) if use_mean_pooling else None
        self.head = nn.Linear(dim_control, num_classes) if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)

        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
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

    def forward_features(self, x):
        x, x_en = self.patch_embed(x)
        B, _, C = x.shape
        x = self.conv_norm(x)   # layer_norm

        if self.model_name == 'neicot_liner':
            noice_nce = self.val_gauss
            nce = 0
            encode_samples = torch.empty((self.timestep, B, C)).float()

            for i in np.arange(0, self.timestep):
                encode_samples[i] = x_en[:,i,:].view(B, C)
            pred = torch.empty((self.timestep, B, C)).float()
            anchor = x.mean(dim = 1)
            # 用全局均值预测局部信息
            for i in np.arange(0, self.timestep):
                conv = self.Wk[i]
                pred[i] = conv(anchor)
            # 用预测出的局部信息与经过池化的数据进行nce计算
            for i in np.arange(0, self.timestep):
                total = torch.mm(encode_samples[i], torch.transpose(pred[i],0,1))   # e.g. size 64*64
                correct = torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0), torch.arange(0, B))) # correct is a tensor
                nce += torch.sum(torch.diag(self.lsoftmax(total)))*noice_nce[i]
            nce /= -1.*C*self.timestep
            acc = 1.*correct.item()/B

        # 添加位置编码再送入Transformer
        x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.model_name == 'neicot_liner':    
            return self.fc_norm(x.mean(1)), nce, acc
        else:
            return self.fc_norm(x.mean(1))
 
    def forward(self, x):
        if self.model_name == 'neicot_liner':
            x, nce, acc = self.forward_features(x)
            x = self.head(x)
            return x, nce, acc
        else:
            x = self.forward_features(x)
            x = self.head(x)            
            return x


@register_model
def vit_3_patch(pretrained=False, **kwargs):
    print("---------:",kwargs)
    embed_dim = kwargs['patch_size']**2
    print("embed_dim",embed_dim)
    model = VisionTransformer(
        patch_size=kwargs['patch_size'], embed_dim=embed_dim, num_classes=kwargs['n_classes'], in_chans=kwargs['n_bands'], depth=kwargs['depth'],
        num_heads=6, model_name=kwargs['model_name123'], sigma = kwargs['sigma'], mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    print("depth_layer:",kwargs['depth'])
    model.default_cfg = _cfg()
    return model

