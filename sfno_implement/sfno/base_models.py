import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_harmonics import RealSHT, InverseRealSHT
from .contractions import contract_dhconv
from torch.cuda import amp

class RealSHT_(nn.Module):
    def __init__(self):
        super().__init__()
        self._SHT_cache = nn.ModuleDict()

    def sht(self, x, modes=None, norm="ortho", grid="equiangular"):
        *_, height, width = x.shape
        modes_height, modes_width = modes

        cache_key = f"{height}_{width}_{modes_height}_{modes_width}_{norm}_{grid}"
        if cache_key not in self._SHT_cache:
            sht = RealSHT(height, width, modes_height, modes_width, grid=grid, norm=norm).to(device=x.device).float()
            self._SHT_cache[cache_key] = sht

        return self._SHT_cache[cache_key](x)

class IRealSHT_(nn.Module):
    def __init__(self):
        super().__init__()
        self._iSHT_cache = nn.ModuleDict()

    def isht(self, x, modes=None, norm="ortho", grid="equiangular"):
        *_, modes_height, modes_width = x.shape
        height, width = modes

        cache_key = f"{height}_{width}_{modes_height}_{modes_width}_{norm}_{grid}"
        if cache_key not in self._iSHT_cache:
            isht = InverseRealSHT(height, width, modes_height, modes_width, grid=grid, norm=norm).to(device=x.device).float()
            self._iSHT_cache[cache_key] = isht

        return self._iSHT_cache[cache_key](x)

class SphericalConv(nn.Module):
    def __init__(
        self,
        forward_transform,
        inverse_transform,
        in_channels,
        out_channels,
        img_size,
        gain=2.,
        bias=False,
        use_embed=False,
        first_layer=False,
        last_layer=False,
        use_nonlinear=False):
        super().__init__()
        
        self.use_embed = use_embed
        self.use_nonlinear = use_nonlinear
        
        # Positional embedding
        if use_embed and first_layer:
            self.pos_embed = nn.Parameter(torch.zeros(1, in_channels, img_size[0], 1)) if use_embed else None
            nn.init.constant_(self.pos_embed, 0.0)

        self.first_layer = first_layer
        self.last_layer = last_layer
        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform
        self.scale_residual = forward_transform.nlat != inverse_transform.nlat or forward_transform.nlon != inverse_transform.nlon

        weight_shape = [out_channels, in_channels, inverse_transform.lmax]
        scale = math.sqrt(gain / in_channels) * torch.ones(inverse_transform.lmax, 2)
        scale[0] *= math.sqrt(2)
        self.weight_1 = nn.Parameter(scale * torch.view_as_real(torch.randn(*weight_shape, dtype=torch.complex64)))
        # if use_nonlinear:
        #     weight_shape = [out_channels, out_channels, inverse_transform.lmax]
        #     scale = math.sqrt(gain / out_channels) * torch.ones(inverse_transform.lmax, 2)
        #     scale[0] *= math.sqrt(2)
        #     self.weight_2 = nn.Parameter(scale * torch.view_as_real(torch.randn(*[out_channels, out_channels, inverse_transform.lmax], dtype=torch.complex64)))
        
        self._contract = contract_dhconv

        if bias:
            self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        if first_layer:
            self.first_forward_transform = RealSHT_()
        if last_layer:
            self.last_inverse_transform = IRealSHT_()

    def forward(self, x, last_image_shape):
        dtype = x.dtype
        x = x.float()
        residual = x

        with amp.autocast(enabled=False):
            if self.first_layer:
                x = self.first_forward_transform.sht(x, (self.inverse_transform.lmax, self.inverse_transform.mmax))
                if self.use_embed:
                    x_embed = self.first_forward_transform.sht(self.pos_embed, (self.inverse_transform.lmax, self.inverse_transform.mmax))
                    x = x + x_embed
            else:
                x = self.forward_transform(x)
            if self.scale_residual:
                residual = self.inverse_transform(x)
            if self.last_layer:
                residual = self.last_inverse_transform.isht(x, last_image_shape)

        x = torch.view_as_real(x)
        x = self._contract(x, self.weight_1)
        if self.use_nonlinear:
            x = F.relu(x)
            # x = self._contract(x, self.weight_2)
        
        x = torch.view_as_complex(x)

        with amp.autocast(enabled=False):
            if self.last_layer:
                x = self.last_inverse_transform.isht(x, last_image_shape)
            else:
                x = self.inverse_transform(x)

        if hasattr(self, "bias"):
            x = x + self.bias
        x = x.type(dtype)
        
        return x, residual

class SFNO_Block(nn.Module):
    def __init__(
            self,
            forward_transform,
            inverse_transform,
            input_dim,
            output_dim,
            img_size,
            mlp_ratio=2.,
            use_embed=False,
            use_act=True,
            use_norm=False,
            use_inner_skip=False,
            use_outer_skip=False,
            first_layer=False,
            last_layer=False,
            use_nonlinear=False):
        super().__init__()

        self.use_act = use_act
        self.use_norm = use_norm
        self.use_inner_skip = use_inner_skip
        self.use_outer_skip = use_outer_skip

        gain_factor = 1.0
        self.filter = SphericalConv(forward_transform, inverse_transform, input_dim, output_dim, img_size, gain=gain_factor, bias=True, 
                                    use_embed=use_embed, first_layer=first_layer, last_layer=last_layer, use_nonlinear=use_nonlinear)
        
        if use_norm:
            self.norm0 = nn.InstanceNorm2d(output_dim, eps=1e-6, affine=True, track_running_stats=False)
        if use_inner_skip:
            self.inner_skip = nn.Conv2d(input_dim, output_dim, 1, 1)
            nn.init.normal_(self.inner_skip.weight, std=math.sqrt(gain_factor/input_dim))
        if use_act:
            self.act_layer = nn.GELU()
        if use_outer_skip:
            self.outer_skip = nn.Conv2d(input_dim, input_dim, 1, 1)
            nn.init.normal_(self.outer_skip.weight, std=math.sqrt(0.5/input_dim))
        if use_norm:
            self.norm1 = nn.InstanceNorm2d(input_dim, eps=1e-6, affine=True, track_running_stats=False)

    def forward(self, x, last_image_shape):
        x, residual = self.filter(x, last_image_shape)

        if self.use_norm:
            x = self.norm0(x)
        if self.use_inner_skip:
            x += self.inner_skip(residual)
        if self.use_act:
            x = self.act_layer(x)
        if self.use_norm:
            x = self.norm1(x)
        if self.use_outer_skip:
            x += self.outer_skip(residual)

        return x

class ComplexReLU(nn.Module):
    """
    Complex-valued variants of the ReLU activation function
    """
    def __init__(self, negative_slope=0., mode="real", bias_shape=None, scale=1.):
        super(ComplexReLU, self).__init__()
        
        # store parameters
        self.mode = mode
        if self.mode in ["modulus", "halfplane"]:
            if bias_shape is not None:
                self.bias = nn.Parameter(scale * torch.ones(bias_shape, dtype=torch.float32))
            else:
                self.bias = nn.Parameter(scale * torch.ones((1), dtype=torch.float32))
        else:
            self.bias = 0

        self.negative_slope = negative_slope
        self.act = nn.LeakyReLU(negative_slope = negative_slope)

    def forward(self, z: torch.Tensor) -> torch.Tensor:

        if self.mode == "cartesian":
            zr = torch.view_as_real(z)
            za = self.act(zr)
            out = torch.view_as_complex(za)

        elif self.mode == "modulus":
            zabs = torch.sqrt(torch.square(z.real) + torch.square(z.imag))
            out = torch.where(zabs + self.bias > 0, (zabs + self.bias) * z / zabs, 0.0)

        elif self.mode == "cardioid":
            out = 0.5 * (1. + torch.cos(z.angle())) * z

        elif self.mode == "real":
            zr = torch.view_as_real(z)
            outr = zr.clone()
            outr[..., 0] = self.act(zr[..., 0])
            out = torch.view_as_complex(outr)
        else:
            raise NotImplementedError
            
        return out