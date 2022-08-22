from turtle import forward
from .mix_transformer import *
from utils import *
from mmseg.ops import resize



class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # x = F.interpolate(x, size=2*x.shape[-1], mode='bilinear', align_corners=True)
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class PatchBlock(nn.Module):
    def __init__(self, img_size, patch_size, depths, in_chans, embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale,
            drop_rate, attn_drop, drop_path_rates, norm_layer, sr_ratio):
        super().__init__()

        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
                
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths))]  # stochastic depth decay rule
        self.block = nn.ModuleList([Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop, drop_path=drop_path_rates[i], norm_layer=norm_layer,
            sr_ratio=sr_ratio)
            for i in range(depths)])
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        B = x.shape[0]
        x, H, W = self.patch_embed(x)
        for i, blk in enumerate(self.block):
            x = blk(x, H, W)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


class PatchFormerBlock(nn.Module):
    def __init__(self, img_size, large_patch, context_padding, patch_size, depths, in_chans, embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale,
            drop_rate, attn_drop, drop_path_rates, norm_layer, sr_ratio, pos_embed, alt):
        super().__init__()

        self.patch_embed = PatchEmbed(
                img_size=large_patch + 2 * context_padding, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
                
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths))]  # stochastic depth decay rule
        self.block = nn.ModuleList([Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop, drop_path=drop_path_rates[i], norm_layer=norm_layer,
            sr_ratio=sr_ratio)
            for i in range(depths)])
        self.norm = norm_layer(embed_dim)
        self.large_patch = large_patch
        self.context_padding = context_padding
        self.patch_size = patch_size
        self.alt = alt

        if pos_embed:
            assert embed_dim is not None
            num_patch = int(img_size/large_patch)**2
            self.register_buffer('pos_embed', torch.zeros(num_patch, 1, embed_dim))
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(num_patch**.5), cls_token=False)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(1))
        else:
            self.pos_embed = None

        if alt:
            self.proj = nn.Conv2d(embed_dim + in_chans, embed_dim, kernel_size=1)


    def forward(self, x):
        img = x
        x = patchify_enlarged(x, self.large_patch, context_padding=self.context_padding)
        B = x.shape[0]

        x, H, W = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed.repeat((img.shape[0], 1, 1))
        for i, blk in enumerate(self.block):
            x = blk(x, H, W)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = unpatchify(x, img.shape[0], context_padding=int(self.context_padding/self.patch_size))

        if self.alt:
            x_scale = resize(input=x, size=img.shape[2:], mode='bilinear', align_corners=False)
            x_cat = torch.cat((img, x_scale), dim=1)
            x = self.proj(x_cat)

        return x


@BACKBONES.register_module()
class PatchTransformer(nn.Module):
    def __init__(self, img_size=224, patch_block_type='patchformer', large_patch=[64,32,16,32], context_padding=[2,2,2,2], patch_sizes=[4, 4, 4, 4], 
                 in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], 
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], pos_embed=False, alt=None):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.patch_block_type = patch_block_type

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cum_depth = np.cumsum(depths)

        self.encoder = nn.ModuleList()
        for i in range(len(depths)):
            drop_path_rate_arr = dpr[0 if i == 0 else cum_depth[i - 1]: cum_depth[i]]
            in_dims = in_chans if i == 0 else embed_dims[i - 1]
            if patch_block_type == 'patchformer':
                block_alt = False if alt is None else alt[i]
                encoder_module = PatchFormerBlock(img_size, large_patch[i], context_padding[i], patch_sizes[i], depths[i], 
                                            in_dims, embed_dims[i], num_heads[i], mlp_ratios[i], qkv_bias, qk_scale,
                                            drop_rate, attn_drop_rate, drop_path_rate_arr, norm_layer, sr_ratios[i], pos_embed, block_alt)
            elif patch_block_type == 'patchblock':
                encoder_module = PatchBlock(img_size, patch_sizes[i], depths[i], 
                                            in_dims, embed_dims[i], num_heads[i], mlp_ratios[i], qkv_bias, qk_scale,
                                            drop_rate, attn_drop_rate, drop_path_rate_arr, norm_layer, sr_ratios[i])
            else:
                raise ValueError('unknow patch_block_type!')
            
            img_size /= patch_sizes[i]
                
            self.encoder.append(encoder_module)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # x: [B, 2, 256, 256]

        # stage 1
        # x = patchify_enlarged(x, self.large_patch, context_padding=self.context_padding)
        # x = self.patch_embed(x)
        # H, W = x.shape[2], x.shape[3]
        # for i, blk in enumerate(self.block1):
        #     x = blk(x, H, W)
        # x = self.norm1(x)
        # x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # outs.append(x)

        # x: [B, 32, 192, 192]
        
        for encoder_module in self.encoder:
            x = encoder_module(x)
            outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x
