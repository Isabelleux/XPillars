import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate


class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


class PillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1
        num_point_features += 3
        num_point_features += 48
        
        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def xblock(self, features):
        new_features = torch.zeros_like(features)

        features_0_16 = features[:, 0:16]   
        features_16_32 = features[:, 16:32]
        features_32_48 = features[:, 32:48]
        features_48_64 = features[:, 48:64] 


        new_features[:, 16:32] = features_16_32
        new_features[:, 32:48] = features_32_48
        prev_features_0_16 = torch.cat((features_0_16[1:], torch.zeros(1, 16, device=features.device)))
        new_features[1:, 0:16] = (0.6*prev_features_0_16[:-1] + 0.4*features_16_32[1:])

        next_features_48_64 = torch.cat((torch.zeros(1, 16, device=features.device), features_48_64[:-1]))
        new_features[:-1, 48:64] = (0.6*next_features_48_64[1:] + 0.4*features_32_48[:-1])

        new_features[0, 48:64] = features_48_64[0]
        new_features[-1, 0:16] = features_0_16[-1]
        
        return new_features

    def position_encoding(self, voxel_features, num_feats=48):
        coords = voxel_features[:, :, :3]

        assert num_feats % 3 == 0, "InputError: num_feats needs to be divisible by 3."
        feats_per_axis = int(num_feats / 3)

        position_enc = torch.cat([
            torch.sin(coords * (10000 ** (2 * (i // 2) / feats_per_axis)))
            if i % 2 == 0
            else
            torch.cos(coords * (10000 ** (2 * (i // 2) / feats_per_axis)))
            for i in range(feats_per_axis)
        ], dim=-1)

        return position_enc
        
    def forward(self, batch_dict, **kwargs):
  
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean
        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
        

        f_variance = torch.var(voxel_features[:, :, :3], dim=1, keepdim=True).expand(-1, voxel_features.shape[1], -1)

        P, M = voxel_features.shape[0], voxel_features.shape[1]
        pos_enc = self.position_encoding(voxel_features, num_feats=96)

        pos_enc = pos_enc.unsqueeze(2).repeat(1, 1, M // pos_enc.shape[1], 1).reshape(P, M, -1)  # [P, M, num_feats]

        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center, f_variance, pos_enc]
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center, f_variance, pos_enc]

        if self.with_distance:
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        voxel_count = features.shape[1]
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
        for pfn in self.pfn_layers:
            features = pfn(features)
        features = features.squeeze()
        features = self.xblock(features)
        batch_dict['pillar_features'] = features
        return batch_dict
