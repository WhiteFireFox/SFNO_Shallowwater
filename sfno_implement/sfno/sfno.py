import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_harmonics import RealSHT, InverseRealSHT
from .base_models import SFNO_Block

class SFNO(nn.Module):
    def __init__(
            self,
            img_size=(128, 256),
            grid="equiangular",
            scale_factor=3,
            in_chans=3,
            out_chans=3,
            embed_dim=256,
            num_layers=4,
            big_skip=False,
            use_embed=True,
            mlp_ratio=2.0,
            use_nonlinear=False):

        super(SFNO, self).__init__()

        self.img_size = img_size
        self.scale_factor = scale_factor
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.big_skip = big_skip
        self.use_embed = use_embed
        self.mlp_ratio = mlp_ratio

        h, w = img_size[0] // scale_factor, img_size[1] // scale_factor
        modes_lat = modes_lon = min(h, w // 2)

        # Encoder
        # encoder_layers = [
        #     nn.Conv2d(in_chans, int(embed_dim * mlp_ratio), 1),
        #     nn.GELU(),
        #     nn.Conv2d(int(embed_dim * mlp_ratio), embed_dim, 1, bias=False)
        # ]
        encoder_layers = [
            nn.Conv2d(in_chans, embed_dim, 1),
            nn.GELU(),
        ]
        self.encoder = nn.Sequential(*encoder_layers)

        # Spectral Transforms
        self.trans_down = RealSHT(*img_size, lmax=modes_lat, mmax=modes_lon, grid=grid).float()
        self.itrans_up = InverseRealSHT(*img_size, lmax=modes_lat, mmax=modes_lon, grid=grid).float()
        self.trans = RealSHT(h, w, lmax=modes_lat, mmax=modes_lon, grid="legendre-gauss").float()
        self.itrans = InverseRealSHT(h, w, lmax=modes_lat, mmax=modes_lon, grid="legendre-gauss").float()

        # Blocks
        self.blocks = nn.ModuleList([
            SFNO_Block(
                self.trans_down if i == 0 else self.trans,
                self.itrans_up if i == num_layers - 1 else self.itrans,
                embed_dim,
                embed_dim,
                img_size=img_size,
                mlp_ratio=mlp_ratio,
                use_embed=use_embed,
                use_act=True,
                use_norm=False,
                use_inner_skip=True,
                use_outer_skip=True,
                first_layer=(i == 0),
                last_layer=(i == num_layers - 1),
                use_nonlinear=use_nonlinear
            ) for i in range(num_layers)
        ])

        # Decoder
        # decoder_layers = [
        #     nn.Conv2d(embed_dim + big_skip * in_chans, int(embed_dim * mlp_ratio), 1),
        #     nn.GELU(),
        #     nn.Conv2d(int(embed_dim * mlp_ratio), out_chans, 1, bias=False)
        # ]
        decoder_layers = [
            nn.Conv2d(embed_dim + big_skip * in_chans, out_chans, 1, bias=False)
        ]
        self.decoder = nn.Sequential(*decoder_layers)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    def forward(self, x):
        last_image_shape = (x.shape[2], x.shape[3])
        residual = x if self.big_skip else None

        x = self.encoder(x)

        for blk in self.blocks:
            x = blk(x, last_image_shape)

        if self.big_skip:
            x = torch.cat((x, residual), dim=1)

        x = self.decoder(x)
        return x
