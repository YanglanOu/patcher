from utils import get_2d_sincos_pos_embed
from .vit import *

@BACKBONES.register_module()
class PatchVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, model_name='vit_large_patch16_384', img_size=384, patch_size=16, in_chans=3, embed_dim=1024, depth=24,
                 num_heads=16, num_classes=19, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.1, attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_cfg=None,
                 pos_embed_interp=False, random_init=False, align_corners=False, pos_embed_type=None, aux=False, **kwargs):
        super(PatchVisionTransformer, self).__init__(**kwargs)
        self.model_name = model_name
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.hybrid_backbone = hybrid_backbone
        self.norm_layer = norm_layer
        self.norm_cfg = norm_cfg
        self.pos_embed_interp = pos_embed_interp
        self.random_init = random_init
        self.align_corners = align_corners
        self.pos_embed_type = pos_embed_type

        self.num_stages = self.depth
        self.out_indices = tuple(range(self.num_stages))

        if self.hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                self.hybrid_backbone, img_size=self.img_size, in_chans=self.in_chans, embed_dim=self.embed_dim)
        else:
            self.patch_embed = PatchEmbed(
                img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        if self.pos_embed_type =='sin_cos':
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim), requires_grad=False)
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))

        self.pos_drop = nn.Dropout(p=self.drop_rate)

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate,
                                                self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                drop=self.drop_rate, attn_drop=self.attn_drop_rate, drop_path=dpr[i], norm_layer=self.norm_layer)
            for i in range(self.depth)])

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(embed_dim, representation_size)
        # self.repr_act = nn.Tanh()
        if aux:
            self.aux_proj = nn.Linear(embed_dim*2, embed_dim)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        # self.apply(self._init_weights)

    def init_weights(self, pretrained=None):
        # nn.init.normal_(self.pos_embed, std=0.02)
        # nn.init.zeros_(self.cls_token)

        if self.pos_embed_type =='sin_cos':
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.img_size/self.patch_size), cls_token=True)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if not self.random_init:
            self.default_cfg = default_cfgs[self.model_name]

            if self.model_name in ['vit_small_patch16_224', 'vit_base_patch16_224']:
                load_pretrained(self, num_classes=self.num_classes, in_chans=self.in_chans, pos_embed_interp=self.pos_embed_interp,
                                num_patches=self.patch_embed.num_patches, align_corners=self.align_corners, filter_fn=self._conv_filter)
            else:
                load_pretrained(self, num_classes=self.num_classes, in_chans=self.in_chans, pos_embed_interp=self.pos_embed_interp,
                                num_patches=self.patch_embed.num_patches, align_corners=self.align_corners)
        else:
            print('Initialize weight randomly')

    @property
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def _conv_filter(self, state_dict, patch_size=16):
        """ convert patch embedding weight from manual patchify + linear proj to conv"""
        out_dict = {}
        for k, v in state_dict.items():
            if 'patch_embed.proj.weight' in k:
                v = v.reshape((v.shape[0], 3, patch_size, patch_size))
            out_dict[k] = v
        return out_dict

    def to_2D(self, x):
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        x = x.transpose(1, 2).reshape(n, c, h, w)
        return x

    def to_1D(self, x):
        n, c, h, w = x.shape
        x = x.reshape(n, c, -1).transpose(1, 2)
        return x

    def forward(self, x, out_pos_embed, aux_x=None):
        B = x.shape[0]

        # aux x
        x = self.patch_embed(x)

        x = x.flatten(2).transpose(1, 2)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        # x: (b h w) (ph pw) e
        # out_pos_embed: (h w) 1 e -> (b h w) 1 e
        x = x + out_pos_embed.repeat((x.shape[0] // out_pos_embed.shape[0], 1, 1))

        if aux_x is not None:
            aux_feat = aux_x[:, 1:, :].reshape(-1, 1, aux_x.shape[-1]).expand_as(x)
            x_aux = torch.cat([x, aux_feat], dim=-1)
            x = self.aux_proj(x_aux)

        x = self.pos_drop(x)

        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
