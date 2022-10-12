import torch
import torch.nn as nn
import torch.nn.functional as F
from src.common import normalize_3d_coordinate

import pdb

class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    Modified based on the implementation of Gaussian Fourier feature mapping.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    """

    def __init__(self, num_input_channels, mapping_size=93, scale=25, learnable=True):
        super().__init__()

        if learnable:
            self._B = nn.Parameter(torch.randn(
                (num_input_channels, mapping_size)) * scale)
        else:
            self._B = torch.randn((num_input_channels, mapping_size)) * scale

    def forward(self, x):
        x = x.squeeze(0)
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(x.dim())
        x = x @ self._B.to(x.device)
        return torch.sin(x)


class Nerf_positional_embedding(torch.nn.Module):
    """
    Nerf positional embedding.

    """

    def __init__(self, multires, log_sampling=True):
        super().__init__()
        self.log_sampling = log_sampling
        self.include_input = True
        self.periodic_fns = [torch.sin, torch.cos]
        self.max_freq_log2 = multires-1
        self.num_freqs = multires
        self.max_freq = self.max_freq_log2
        self.N_freqs = self.num_freqs

    def forward(self, x):
        x = x.squeeze(0)
        assert x.dim() == 2, 'Expected 2D input (got {}D input)'.format(
            x.dim())

        if self.log_sampling:
            freq_bands = 2.**torch.linspace(0.,
                                            self.max_freq, steps=self.N_freqs)
        else:
            freq_bands = torch.linspace(
                2.**0., 2.**self.max_freq, steps=self.N_freqs)
        output = []
        if self.include_input:
            output.append(x)
        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                output.append(p_fn(x * freq))
        ret = torch.cat(output, dim=1)
        return ret


class DenseLayer(nn.Linear):
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", *args, **kwargs) -> None:
        self.activation = activation
        super().__init__(in_dim, out_dim, *args, **kwargs)

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(
            self.weight, gain=torch.nn.init.calculate_gain(self.activation))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class Same(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.squeeze(0)
        return x


class MLP(nn.Module):
    """
    Decoder. Point coordinates not only used in sampling the feature grids, but also as MLP input.

    Args:
        name (str): name of this decoder.
        dim (int): input dimension.
        c_dim (int): feature dimension.
        hidden_size (int): hidden size of Decoder network.
        n_blocks (int): number of layers.
        leaky (bool): whether to use leaky ReLUs.
        sample_mode (str): sampling feature strategy, bilinear|nearest.
        color (bool): whether or not to output color.
        skips (list): list of layers to have skip connections.
        grid_len (float): voxel length of its corresponding feature grid.
        pos_embedding_method (str): positional embedding method.
        concat_feature (bool): whether to get feature from middle level and concat to the current feature.
    """

    def __init__(self, name='', dim=3, c_dim=128,
                 hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear',
                 color=False, skips=[2], grid_len=0.16, pos_embedding_method='fourier', concat_feature=False, feature_fusion=False, n_frames=10):
        super().__init__()
        self.name = name
        self.color = color
        self.no_grad_feature = False
        self.c_dim = c_dim
        self.grid_len = grid_len
        self.concat_feature = concat_feature
        self.n_blocks = n_blocks
        self.skips = skips
        self.feature_fusion = feature_fusion
        self.n_frames = n_frames

        if c_dim != 0:
            if 'color' in name:
                # c_dim += (128 + 3 + 3)
                # c_dim += (128 + 3 * self.n_frames)
                c_dim += (3 * self.n_frames)
                # c_dim += 3
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        if pos_embedding_method == 'fourier':
            embedding_size = 93
            self.embedder = GaussianFourierFeatureTransform(
                dim, mapping_size=embedding_size, scale=25)
        elif pos_embedding_method == 'same':
            embedding_size = 3
            self.embedder = Same()
        elif pos_embedding_method == 'nerf':
            if 'color' in name:
                multires = 15
                self.embedder = Nerf_positional_embedding(
                    multires, log_sampling=True)
                # self.embedderdir = Nerf_positional_embedding(
                #     multires, log_sampling=True)
            else:
                multires = 15
                self.embedder = Nerf_positional_embedding(
                    multires, log_sampling=False)
            embedding_size = multires*6+3
        elif pos_embedding_method == 'fc_relu':
            embedding_size = 93
            self.embedder = DenseLayer(dim, embedding_size, activation='relu')

        # if 'color' in name: 
        #     embedding_size *= 2

        self.pts_linears = nn.ModuleList(
            [DenseLayer(embedding_size, hidden_size, activation="relu")] +
            [DenseLayer(hidden_size, hidden_size, activation="relu") if i not in self.skips
             else DenseLayer(hidden_size + embedding_size, hidden_size, activation="relu") for i in range(n_blocks-1)])

        if self.color:
            self.output_linear = DenseLayer(
                hidden_size, 4, activation="linear")
        else:
            self.output_linear = DenseLayer(
                hidden_size, 1, activation="linear")
        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode

    def sample_grid_feature(self, p, c):
        c = c.to(f'cuda:{p.get_device()}')
        p_nor = normalize_3d_coordinate(p.clone(), self.bound)
        p_nor = p_nor.unsqueeze(0)
        vgrid = p_nor[:, :, None, None].float()
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True,
                        mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    def forward(self, p, c_grid=None, color_c_grid=None, last_feat=False):
        if self.c_dim != 0:
            c = self.sample_grid_feature(
                p, c_grid['grid_' + self.name])
            c = c.transpose(1, 2).squeeze(0)

            if self.concat_feature:
                # only happen to fine decoder, get feature from middle level and concat to the current feature
                with torch.no_grad():
                    c_middle = self.sample_grid_feature(
                        p, c_grid['grid_middle']).transpose(1, 2).squeeze(0)
                c = torch.cat([c, c_middle], dim=1)

            if 'color' in self.name:
                # feat_c = self.sample_grid_feature(
                #     p, feat_c_grid)
                with torch.no_grad():
                    color_c = self.sample_grid_feature(
                        p, color_c_grid)
                # feat_c = feat_c.transpose(1, 2).squeeze(0)
                # c = torch.cat([c, feat_c], dim=-1)
                color_c = color_c.transpose(1, 2).squeeze(0)
                c = torch.cat([c, color_c], dim=-1)
                del color_c

        p = normalize_3d_coordinate(p, self.bound).float()

        embedded_pts = self.embedder(p)
        # if 'color' in self.name:
        #     embedded_dir = self.embedderdir(dir)
        #     embedded_pts = torch.cat([embedded_pts, embedded_dir], dim=-1)
        h = embedded_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if self.c_dim != 0:
                fc_c = self.fc_c[i](c)
                h = h + fc_c
            if i in self.skips:
                h = torch.cat([embedded_pts, h], -1)
        
        out = self.output_linear(h)
        
        if not self.color:
            out = out.squeeze(-1)
        
        if last_feat:
            return out, h
        return out


class MLP_color(nn.Module):
    """
    Decoder. Point coordinates not only used in sampling the feature grids, but also as MLP input.

    Args:
        name (str): name of this decoder.
        dim (int): input dimension.
        c_dim (int): feature dimension.
        hidden_size (int): hidden size of Decoder network.
        n_blocks (int): number of layers.
        leaky (bool): whether to use leaky ReLUs.
        sample_mode (str): sampling feature strategy, bilinear|nearest.
        color (bool): whether or not to output color.
        skips (list): list of layers to have skip connections.
        grid_len (float): voxel length of its corresponding feature grid.
        pos_embedding_method (str): positional embedding method.
        concat_feature (bool): whether to get feature from middle level and concat to the current feature.
    """

    def __init__(self, name='', dim=3, c_dim=128,
                 hidden_size=256, n_blocks=3, sample_mode='bilinear',
                 skips=[1], grid_len=0.16, pos_embedding_method='fourier', n_frames=10):
        super().__init__()
        self.name = name
        self.no_grad_feature = False
        self.c_dim = c_dim
        self.skips = skips
        self.grid_len = grid_len
        self.n_blocks = n_blocks
        self.n_frames = n_frames
        
        if pos_embedding_method == 'fourier':
            embedding_size = 93
            self.embedder = GaussianFourierFeatureTransform(
                dim, mapping_size=embedding_size, scale=25)
        elif pos_embedding_method == 'same':
            embedding_size = 3
            self.embedder = Same()
        elif pos_embedding_method == 'nerf':
            multires = 15
            self.embedder = Nerf_positional_embedding(
                multires, log_sampling=True)
            self.embedderdir = Nerf_positional_embedding(
                multires, log_sampling=True)
            embedding_size = multires*6+3
        elif pos_embedding_method == 'fc_relu':
            embedding_size = 93
            self.embedder = DenseLayer(dim, embedding_size, activation='relu')


        # option 1: geo_feat + points
        # self.bottleneck_linear_1 = DenseLayer(hidden_size + embedding_size, hidden_size, activation='relu')
        # option 2: geo_feat + points + color_feature_grid
        self.bottleneck_linear_1 = DenseLayer(hidden_size + c_dim + embedding_size, hidden_size, activation='relu')
        
        # feat + colors + dir
        # self.pts_linears = nn.ModuleList(
        #     [DenseLayer(hidden_size + (3 * self.n_frames) + embedding_size, hidden_size, activation="relu")] +
        #     [DenseLayer(hidden_size, hidden_size, activation="relu") if i not in self.skips
        #      else DenseLayer(hidden_size + embedding_size, hidden_size, activation="relu") for i in range(n_blocks-1)])
        # feat + colors
        self.pts_linears = nn.ModuleList(
            [DenseLayer(hidden_size + (3 * self.n_frames), hidden_size, activation="relu")] +
            [DenseLayer(hidden_size, hidden_size, activation="relu") if i not in self.skips
             else DenseLayer(hidden_size + (3 * self.n_frames), hidden_size, activation="relu") for i in range(n_blocks-1)])

        self.output_linear = DenseLayer(
                hidden_size, 3, activation="linear")

        self.sample_mode = sample_mode

    def sample_grid_feature(self, p, c):
        c = c.to(f'cuda:{p.get_device()}')
        p_nor = normalize_3d_coordinate(p.clone(), self.bound)
        p_nor = p_nor.unsqueeze(0)
        vgrid = p_nor[:, :, None, None].float()
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True,
                        mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    def forward(self, p, dir, geo_feat, c_grid=None, color_c_grid=None):
        with torch.no_grad():
            color_c = self.sample_grid_feature(
                p, color_c_grid)
        color_c = color_c.transpose(1, 2).squeeze(0)

        p = normalize_3d_coordinate(p, self.bound).float()

        embedded_pts = self.embedder(p)
        embedded_dir = self.embedderdir(dir)
        
        # option 1: geo_feat + points
        # h = torch.cat([geo_feat, embedded_pts], dim=-1)
        # option 2: geo_feat + points + color_feature_grid
        c = self.sample_grid_feature(
            p, c_grid['grid_' + self.name])
        c = c.transpose(1, 2).squeeze(0)
        h = torch.cat([geo_feat, c, embedded_pts], dim=-1)
        
        h = self.bottleneck_linear_1(h)
        
        # h = torch.cat([h, color_c, embedded_dir], dim=-1)
        h = torch.cat([h, color_c], dim=-1)
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                # h = torch.cat([embedded_dir, h], dim=-1)
                h = torch.cat([h, color_c], dim=-1)
        
        out = self.output_linear(h)
        return out

class MLP_no_xyz(nn.Module):
    """
    Decoder. Point coordinates only used in sampling the feature grids, not as MLP input.

    Args:
        name (str): name of this decoder.
        dim (int): input dimension.
        c_dim (int): feature dimension.
        hidden_size (int): hidden size of Decoder network.
        n_blocks (int): number of layers.
        leaky (bool): whether to use leaky ReLUs.
        sample_mode (str): sampling feature strategy, bilinear|nearest.
        color (bool): whether or not to output color.
        skips (list): list of layers to have skip connection.
        grid_len (float): voxel length of its corresponding feature grid.
    """

    def __init__(self, name='', dim=3, c_dim=128,
                 hidden_size=256, n_blocks=5, leaky=False,
                 sample_mode='bilinear', color=False, skips=[2], grid_len=0.16, feature_fusion=False):
        super().__init__()
        self.name = name
        self.no_grad_feature = False
        self.color = color
        self.grid_len = grid_len
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.skips = skips
        self.feature_fusion = feature_fusion

        self.pts_linears = nn.ModuleList(
            [DenseLayer(hidden_size, hidden_size, activation="relu")] +
            [DenseLayer(hidden_size, hidden_size, activation="relu") if i not in self.skips
             else DenseLayer(hidden_size + c_dim, hidden_size, activation="relu") for i in range(n_blocks-1)])

        if self.color:
            self.output_linear = DenseLayer(
                hidden_size, 4, activation="linear")
        else:
            self.output_linear = DenseLayer(
                hidden_size, 1, activation="linear")
        
        if self.feature_fusion and self.name != '':
            self.fusion_linear = DenseLayer(
                1, c_dim, activation="relu")

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode

    def sample_grid_feature(self, p, grid_feature):
        # print('     sample_grid_feature(MLP_no_xyz)')
        # print('     p:', p.shape)
        # print('     grid_feature:', grid_feature.shape)
        grid_feature = grid_feature.to(f'cuda:{p.get_device()}')
        p_nor = normalize_3d_coordinate(p.clone(), self.bound)
        # print('     p_nor:', p_nor.shape)
        p_nor = p_nor.unsqueeze(0)
        # print('     p_nor(unsqueeze):', p_nor.shape)
        vgrid = p_nor[:, :, None, None].float()
        # print('     vgrid:', vgrid.shape)
        c = F.grid_sample(grid_feature, vgrid, padding_mode='border',
                        align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        # print('     c:', c.shape)
        return c

    def forward(self, p, c_grid, **kwargs):
        # print('-'*30)
        # print('MLP_no_xyz')
        
        # print('p:', p.shape)
        # print('c_grid:', c_grid['grid_' + self.name].shape)
        
        c = self.sample_grid_feature(
            p, c_grid['grid_' + self.name])
        # print('c(before):', c.shape)
        c = c.transpose(1, 2).squeeze(0)
        if self.feature_fusion and self.name != '':
            c = self.fusion_linear(c)
        # print('c(after):', c.shape)
        
        h = c
        for i, l in enumerate(self.pts_linears):
            # print('----')
            # print('     i:', i)
            # print('     h', h.shape)
            h = self.pts_linears[i](h)
            # print('     h(pts_linears)', h.shape)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([c, h], -1)
            #     print('     h(skip):', h.shape)
            # print('     h:', h.shape)
        out = self.output_linear(h)
        # print('out:', out.shape)
        if not self.color:
            out = out.squeeze(-1)
            # print('out(not color):', out.shape)
        # print('-'*30)
        return out


class NICE(nn.Module):
    """    
    Neural Implicit Scalable Encoding.

    Args:
        dim (int): input dimension.
        c_dim (int): feature dimension.
        coarse_grid_len (float): voxel length in coarse grid.
        middle_grid_len (float): voxel length in middle grid.
        fine_grid_len (float): voxel length in fine grid.
        color_grid_len (float): voxel length in color grid.
        hidden_size (int): hidden size of decoder network
        coarse (bool): whether or not to use coarse level.
        pos_embedding_method (str): positional embedding method.
    """

    def __init__(self, dim=3, c_dim=32,
                 coarse_grid_len=2.0,  middle_grid_len=0.16, fine_grid_len=0.16,
                 color_grid_len=0.16, hidden_size=32, coarse=False, pos_embedding_method='fourier', feature_fusion=False,
                 n_frames=10):
        super().__init__()

        if coarse:
            self.coarse_decoder = MLP_no_xyz(
                name='coarse', dim=dim, c_dim=c_dim, color=False, hidden_size=hidden_size, grid_len=coarse_grid_len, feature_fusion=feature_fusion)

        self.middle_decoder = MLP(name='middle', dim=dim, c_dim=c_dim, color=False,
                                  skips=[2], n_blocks=5, hidden_size=hidden_size,
                                  grid_len=middle_grid_len, pos_embedding_method=pos_embedding_method, feature_fusion=feature_fusion)
        self.fine_decoder = MLP(name='fine', dim=dim, c_dim=c_dim*2, color=False,
                                skips=[2], n_blocks=5, hidden_size=hidden_size,
                                grid_len=fine_grid_len, concat_feature=True, pos_embedding_method=pos_embedding_method, feature_fusion=feature_fusion)
        # self.color_decoder = MLP(name='color', dim=dim, c_dim=c_dim, color=True,
        #                          skips=[2], n_blocks=5, hidden_size=hidden_size,
        #                          grid_len=color_grid_len, pos_embedding_method=pos_embedding_method, feature_fusion=feature_fusion, n_frames=n_frames)
        self.color_decoder = MLP_color(name='color', dim=dim, c_dim=c_dim,
                                       skips=[1], n_blocks=5, hidden_size=hidden_size,
                                       grid_len=color_grid_len, pos_embedding_method=pos_embedding_method, n_frames=n_frames)


    def forward(self, p, dir, c_grid, stage='middle', color_c=None, **kwargs):
        """
            Output occupancy/color in different stage.
        """
        # print('='*40)
        # print('Rendering Net')
        # print('Stage:', stage)
        
        device = f'cuda:{p.get_device()}'
        if stage == 'coarse':
            occ = self.coarse_decoder(p, c_grid)
            occ = occ.squeeze(0)
            raw = torch.zeros(occ.shape[0], 4).to(device).float()
            raw[..., -1] = occ
            # print('points:', p.shape)
            # print('Feature Grid:', c_grid['grid_'+stage].shape)
            # print('Occ:', occ.shape)
            # print('Raw:', raw.shape)
            return raw
        elif stage == 'middle':
            middle_occ = self.middle_decoder(p, c_grid)
            middle_occ = middle_occ.squeeze(0)
            raw = torch.zeros(middle_occ.shape[0], 4).to(device).float()
            raw[..., -1] = middle_occ
            # print('points:', p.shape)
            # print('Feature Grid:', c_grid['grid_'+stage].shape)
            # print('Occ:', middle_occ.shape)
            # print('Raw:', raw.shape)
            return raw
        elif stage == 'fine':
            fine_occ = self.fine_decoder(p, c_grid)
            raw = torch.zeros(fine_occ.shape[0], 4).to(device).float()
            middle_occ = self.middle_decoder(p, c_grid)
            middle_occ = middle_occ.squeeze(0)
            raw[..., -1] = fine_occ+middle_occ
            # print('points:', p.shape)
            # print('Feature Grid:', c_grid['grid_'+stage].shape)
            # print('Fine Occ:', fine_occ.shape)
            # print('Middle Occ:', middle_occ.shape)
            # print('Raw:', raw.shape)
            return raw
        elif stage == 'color':
            # occ
            fine_occ, geo_feat = self.fine_decoder(p, c_grid, last_feat=True)
            middle_occ = self.middle_decoder(p, c_grid)
            middle_occ = middle_occ.squeeze(0)
            
            raw = torch.zeros(fine_occ.shape[0], 4).to(device).float()
            color = self.color_decoder(p, dir, geo_feat, c_grid, color_c)
            raw[..., :-1] = color
            raw[..., -1] = fine_occ+middle_occ
            
            # sdf / density
            # raw = self.color_decoder(p, c_grid, color_c)
            # middle_occ = self.middle_decoder(p, c_grid)
            # middle_occ = middle_occ.squeeze(0)
            # raw[..., -1] = raw[..., -1] + middle_occ
            
            
            # print('points:', p.shape)
            # print('Feature Grid:', c_grid['grid_'+stage].shape)
            # print('Fine Occ:', fine_occ.shape)
            # print('Middle Occ:', middle_occ.shape)
            # print('Raw:', raw.shape)
            return raw