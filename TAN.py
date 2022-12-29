# Uploads will be made after publication.
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import torch.nn as nn
import torch
import math

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm,self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super(FeedForward,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super(Attention,self).__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super(Transformer,self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class TA(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super(TA,self).__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = dim * patch_height * patch_width

        if patch_size !=1:
            self.h = int(math.sqrt(num_patches))
            self.repeat_h = int(image_size//math.sqrt(num_patches))
        else:
            self.h = image_height

        self.patch_size = patch_size

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim,dim) if patch_size !=1 else nn.Identity()
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, num_patches, 1))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim+1, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        if self.pool == "mean":
            self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, img):

        x = self.to_patch_embedding(img)
        b, n, d = x.shape

        x = x * F.sigmoid(self.pos_embedding)  # 通道注意力权重分配

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=2)   # 像素注意力tokens

        x = self.transformer(x)

        x = self.pooling(x) if self.pool=='mean' else x[:, :, None, 0]

        if self.patch_size !=1:
            x = rearrange(x, 'b (h w) c -> b c h w', h=self.h)
            x = repeat(x, 'b c h w -> b c (h h1) (w w1)', h1=self.repeat_h,w1=self.repeat_h)
        else:
            x = rearrange(x, 'b (h w) c -> b c h w', h=self.h)
        return x

class Double13(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Double13, self).__init__()
        self.conva = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
        self.convb = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
    def forward(self, input):
        x1 = self.conva(input)
        x2 = self.convb(input)
        return x1 + x2

class CConv(nn.Module): # 轮廓卷积
    def __init__(self, in_ch, out_ch):
        super(CConv, self).__init__()
        self.conv = nn.Sequential(
            Double13(in_ch, out_ch),
            nn.Conv2d(out_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            )
    def forward(self, input):
        return self.conv(input)

class TAN_blocks(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, dim_head=64, dropout=0., emb_dropout=0.,in_channels=None):
        super(TAN_blocks,self).__init__()

        if in_channels != dim:
            dim_trans = in_channels
        else:
            dim_trans = dim

        self.trans = TA(
        image_size=image_size,
        patch_size=patch_size,
        dim=dim_trans,
        depth=depth,
        heads=heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        emb_dropout=emb_dropout,
        dim_head=dim_head
    )
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels,dim, 3, 1,1),
            nn.ReLU(True),

            CConv(dim,dim)
        )

    def forward(self, x):
        x1 = self.cnn(x)
        x2 = self.trans(x)
        x = x1*F.sigmoid(x2)
        return x

# Transformer Attention network（TAN）
class TAN(nn.Module):
    def __init__(self, in_channels,class_num,image_size, patch_size, dim, depth, heads, mlp_dim, dim_head=8, dropout=0., emb_dropout=0.):
        super(TAN,self).__init__()

        self.deh = nn.Sequential(
            nn.Conv2d(in_channels,dim,1)
        )

        self.cnn_trans = nn.Sequential(
            TAN_blocks(image_size, patch_size, dim, depth, heads, mlp_dim, dropout=dropout,
                                     emb_dropout=emb_dropout,dim_head=dim_head,in_channels=dim),
            TAN_blocks(image_size, patch_size, dim, depth, heads, mlp_dim, dropout=dropout,
                                     emb_dropout=emb_dropout,in_channels=dim),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(dim,32),
            nn.ReLU(),

            nn.Linear(32,class_num)
        )

    def forward(self, x):
        x = self.deh(x)
        x = self.cnn_trans(x)
        x = self.pool(x).squeeze()
        x = self.fc(x)
        return x

if __name__=="__main__":
    from torchsummary import summary

    model_vit = TAN(
        in_channels=112,
        class_num=4,
        image_size=9,
        patch_size=1,
        dim=64,
        depth=1,
        heads=1,
        mlp_dim=32,
    )

    summary(model_vit, (112, 9, 9),device="cpu")
