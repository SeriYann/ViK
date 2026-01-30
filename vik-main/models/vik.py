import copy
import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple



def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'vik_small': _cfg(crop_pct=0.9),
    'vik_base': _cfg(crop_pct=0.9),
}

class RBFLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_min: float = -2,
        grid_max: float = 2,
        num_grids: int = 8,
        spline_weight_init_scale: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_grids = num_grids


        base_grid = torch.linspace(grid_min, grid_max, num_grids)
        mu_init = base_grid.repeat(in_features)           

        sigma_init = torch.full((in_features * num_grids,), (base_grid[1] - base_grid[0]).abs() + 1e-6)


        self.mu = nn.Parameter(mu_init)
        self.log_sigma = nn.Parameter(sigma_init.log())


        self.spline_weight = nn.Parameter(
            torch.randn(in_features * num_grids, out_features) * spline_weight_init_scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, F = x.shape
        G = self.num_grids

        x_flat = x.reshape(B * S, F)
        mu = self.mu.view(1, F, G).to(dtype=x.dtype, device=x.device)
        sigma = self.log_sigma.exp().view(1, F, G).to(dtype=x.dtype, device=x.device) + 1e-6


        x_exp = x_flat.unsqueeze(-1)
        basis = torch.exp(-0.5 * ((x_exp - mu) / sigma) ** 2)
        basis = basis.reshape(B * S, F * G)

        out = basis @ self.spline_weight
        return out.view(B, S, self.out_features)


class RBFKANLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        grid_min: float = -2,
        grid_max: float = 2,
        num_grids: int = 8,
        use_base_update: bool = True,
        base_activation=nn.SiLU(), 
        spline_weight_init_scale: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_base_update = use_base_update

        self.base_activation = type(base_activation)() if isinstance(base_activation, nn.Module) else nn.SiLU()

        self.rbf_linear = RBFLinear(
            input_dim, output_dim,
            grid_min=grid_min, grid_max=grid_max,
            num_grids=num_grids,
            spline_weight_init_scale=spline_weight_init_scale
        )
        if use_base_update:
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.rbf_linear(x)
        if self.use_base_update:
            y = y + self.base_linear(self.base_activation(x))
        return y


class MultiPatchRBFKAN(nn.Module):
    def __init__(
        self,
        patch_size: int = 7,
        input_channels: int = 1,
        image_size=None,
        use_global_linear: bool = True
    ):
        super().__init__()
        self.ps = patch_size
        f = patch_size * patch_size
        self.patch_mixer = RBFKANLayer(
            input_dim=f, output_dim=f, num_grids=8,
            use_base_update=True, base_activation=nn.SiLU()
        )

        self.use_separable = True
        self.sep_kernel = 3
        self._sep_built_for_c = None
        self.dw_h = None
        self.dw_w = None
        self.reweight_head = None


        self.use_global_linear = use_global_linear
        self.global_rank_default = 64 
        self._lowrank_built_for_hw = None
        self.lin1 = None
        self.lin2 = None
        self.H = None
        self.W = None


    def initialize(self, H: int, W: int, device=None):
        self.H, self.W = H, W 

    def _build_separable(self, C: int, device):
        if self._sep_built_for_c == C:
            return
        k = self.sep_kernel
        self.dw_h = nn.Conv2d(C, C, kernel_size=(k, 1), padding=(k // 2, 0), groups=C).to(device)
        self.dw_w = nn.Conv2d(C, C, kernel_size=(1, k), padding=(0, k // 2), groups=C).to(device)
        self.reweight_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C, max(8, C // 4), 1), nn.GELU(),
            nn.Conv2d(max(8, C // 4), 2, 1),
            nn.Softmax(dim=1)
        ).to(device)
        self._sep_built_for_c = C

 
    def _build_lowrank(self, H: int, W: int, device):
        if self._lowrank_built_for_hw == (H, W):
            return
        N = H * W
        r = min(self.global_rank_default, N)
        self.lin1 = nn.Linear(N, r, bias=False).to(device)
        self.lin2 = nn.Linear(r, N, bias=False).to(device)
        self._lowrank_built_for_hw = (H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ps = self.ps
        assert H % ps == 0 and W % ps == 0, f"H/W has to be devided by patch_size={ps}"


        patches = x.unfold(2, ps, ps).unfold(3, ps, ps) 
        Hn, Wn = H // ps, W // ps
        tokens = patches.contiguous().view(B * C, Hn * Wn, ps * ps)

        mixed = self.patch_mixer(tokens)
        mixed = mixed + tokens


        mixed = mixed.view(B, C, Hn, Wn, ps, ps).permute(0, 1, 2, 4, 3, 5).contiguous()
        y = mixed.view(B, C, H, W)


        if self.use_separable:
            self._build_separable(C, x.device)
            y_h = self.dw_h(y)
            y_w = self.dw_w(y)
            w = self.reweight_head(y)
            y = w[:, 0:1] * y_h + w[:, 1:2] * y_w

        if self.use_global_linear:
            self._build_lowrank(H, W, x.device)
            y_flat = y.view(B * C, -1)
            y_global = self.lin2(self.lin1(y_flat)).view(B, C, H, W)
            y = y + y_global

        return y




class PatchEmbed(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv. 
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """
    def __init__(self, patch_size=16, stride=16, padding=0, 
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, 
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class LayerNormChannel(nn.Module):
    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x


class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, 
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MPRBFBlock(nn.Module):
    def __init__(self, dim, pool_size=3, mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop=0., drop_path=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = MultiPatchRBFKAN()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def basic_blocks(dim, index, layers, 
                 pool_size=3, mlp_ratio=4., 
                 act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop_rate=.0, drop_path_rate=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5):

    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
            block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(MPRBFBlock(
            dim, pool_size=pool_size, mlp_ratio=mlp_ratio, 
            act_layer=act_layer, norm_layer=norm_layer, 
            drop=drop_rate, drop_path=block_dpr, 
            use_layer_scale=use_layer_scale, 
            layer_scale_init_value=layer_scale_init_value, 
            ))
    blocks = nn.Sequential(*blocks)

    return blocks


class ViK(nn.Module):
    def __init__(self, layers, embed_dims=None, 
                 mlp_ratios=None, downsamples=None, 
                 pool_size=3, 
                 norm_layer=GroupNorm, act_layer=nn.GELU, 
                 num_classes=1000,
                 in_patch_size=7, in_stride=4, in_pad=2, 
                 down_patch_size=3, down_stride=2, down_pad=1, 
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5, 
                 pretrained=None, 
                 **kwargs):

        super().__init__()


        self.num_classes = num_classes


        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad, 
            in_chans=3, embed_dim=embed_dims[0])

        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers, 
                                 pool_size=pool_size, mlp_ratio=mlp_ratios[i],
                                 act_layer=act_layer, norm_layer=norm_layer, 
                                 drop_rate=drop_rate, 
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale, 
                                 layer_scale_init_value=layer_scale_init_value)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i+1]:
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size, stride=down_stride, 
                        padding=down_pad, 
                        in_chans=embed_dims[i], embed_dim=embed_dims[i+1]
                        )
                    )

        self.network = nn.ModuleList(network)


        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(
            embed_dims[-1], num_classes) if num_classes > 0 \
            else nn.Identity()

        self.apply(self.cls_init_weights)


    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
         

    def get_classifier(self):
        return self.head


    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        for idx, block in enumerate(self.network):
            x = block(x)
        return x

    def forward(self, x):
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)
        x = self.norm(x)
        cls_out = self.head(x.mean([-2, -1]))
        # for image classification
        return cls_out


model_urls = {}


@register_model
def vik_small(pretrained=False, **kwargs):
    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = ViK(
        layers, embed_dims=embed_dims, 
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        **kwargs)
    model.default_cfg = default_cfgs['vik_small']
    if pretrained:
        url = model_urls['vik_small']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model


@register_model
def vik_base(pretrained=False, **kwargs):
    layers = [4, 4, 12, 4]
    embed_dims = [64, 128, 320, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    model = ViK(
        layers, embed_dims=embed_dims, 
        mlp_ratios=mlp_ratios, downsamples=downsamples, 
        **kwargs)
    model.default_cfg = default_cfgs['vik_base']
    if pretrained:
        url = model_urls['vik_base']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint)
    return model
