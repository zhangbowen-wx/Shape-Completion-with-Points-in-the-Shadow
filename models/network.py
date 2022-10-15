import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pn2_utils import PointNet_SA_Module_KNN, MLP_Res, MLP_CONV, Transformer
from models.skip_transformer import SkipTransformer
from models.utils import fps_subsample_with_dist
from models import tools


class PN_Encoder(nn.Module):
    def __init__(self, out_dim=1024):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(3,128,1)
        self.conv2 = torch.nn.Conv1d(128,256,1)
        self.conv3 = torch.nn.Conv1d(512,512,1)
        self.conv4 = torch.nn.Conv1d(512,out_dim,1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        global_feature_1 = torch.max(x, dim=2, keepdim=True)[0].repeat(1, 1, x.shape[2])
        x = torch.cat([x, global_feature_1], dim=1)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        global_feature_2 = torch.max(x, dim=2, keepdim=True)[0] # [batch_size, out_dim, 1]
        return global_feature_2 


class PN2_Encoder(nn.Module):
    def __init__(self, out_dim=1024):
        """Encoder that encodes information of partial point cloud
        """
        super().__init__()
        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.transformer_1 = Transformer(128, dim=64)
        self.sa_module_2 = PointNet_SA_Module_KNN(128, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.transformer_2 = Transformer(256, dim=64)
        self.sa_module_3 = PointNet_SA_Module_KNN(None, None, 256, [512, out_dim], group_all=True, if_bn=False)

    def forward(self, point_cloud):
        """
        Args:
             point_cloud: b, 3, n

        Returns:
            l3_points: (B, out_dim, 1)
        """
        l0_xyz = point_cloud
        l0_points = point_cloud

        l1_xyz, l1_points, idx1 = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
        l1_points = self.transformer_1(l1_points, l1_xyz)
        l2_xyz, l2_points, idx2 = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 512)
        l2_points = self.transformer_2(l2_points, l2_xyz)
        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 1), (B, out_dim, 1)

        return l3_points


class VP_Model(nn.Module):
    def __init__(self, global_feat_dim=1024, out_mode="UnitVector"):
        super().__init__()
        self.out_mode = out_mode
        if(out_mode == "Quaternion"):
            self.out_channel = 4
        elif (out_mode  == "ortho6d"):
            self.out_channel = 6
        elif (out_mode  == "svd9d"):
            self.out_channel = 9
        elif (out_mode  == "10d"):
            self.out_channel = 10
        elif out_mode == 'euler':
            self.out_channel = 3
        elif out_mode == 'axisangle':
            self.out_channel = 4
        elif out_mode == 'UnitVector':
            self.out_channel = 3
        else:
            raise NotImplementedError
        self.global_feat_dim = global_feat_dim
        self.encoder = PN2_Encoder(global_feat_dim)
        self.fc1 = nn.Linear(global_feat_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, self.out_channel)
    def forward(self, input, if_normlize=True):
        """
        Args:
            input: (b, n, 3)
        Return:
            view_point: (b, 3)
        """
        input = input.permute(0,2,1).contiguous()
        global_feat = self.encoder(input)
        x = F.relu(self.fc1(global_feat.view(-1, self.global_feat_dim)))
        x = F.relu(self.fc2(x))
        out_nd = torch.tanh(self.fc3(x)) # (b, nd)
        if self.out_mode=='UnitVector':
            if if_normlize:
                view_point = out_nd / torch.norm(out_nd, dim=1, keepdim=True)
                return view_point
            else:
                view_point = out_nd
                return view_point
        else:
            if(self.out_mode == "Quaternion"):
                out_rmat = tools.compute_rotation_matrix_from_quaternion(out_nd) #b*3*3
            elif(self.out_mode=="ortho6d"):
                out_rmat = tools.compute_rotation_matrix_from_ortho6d(out_nd) #b*3*3
            elif(self.out_mode=="svd9d"):
                out_rmat = tools.symmetric_orthogonalization(out_nd)  # b*3*3
            elif (self.out_mode == "10d"):
                out_rmat = tools.compute_rotation_matrix_from_10d(out_nd)  # b*3*3
            elif (self.out_mode == "euler"):
                out_rmat = tools.compute_rotation_matrix_from_euler(out_nd)  # b*3*3
            elif (self.out_mode == "axisangle"):
                out_rmat = tools.compute_rotation_matrix_from_axisAngle(out_nd)  # b*3*3
            y = torch.tensor([[[0.0, 1.0, 0.0]]], dtype=torch.float32).repeat(input.shape[0], 1, 1).cuda() # (b, 1, 3)
            view_point = torch.bmm(y, out_rmat) # (b, 1, 3)
            view_point = view_point.view(-1, 3)
            return view_point


class Adjusted_VP_Model(nn.Module):
    def __init__(self, global_feat_dim=1024, out_mode="UnitVector"):
        super().__init__()
        self.vp1 = VP_Model(global_feat_dim, out_mode)
        self.rt = SkipTransformer(in_channel=3, dim=64)
        self.vp2 = VP_Model(global_feat_dim, out_mode)

    def forward(self, input):
        """
        Args:
            input: (b, n, 3)
        Return:
            view_point: (b, 3)
            adjusted_view_point: (b, 3)
        """
        view_point = self.vp1(input)
        input = input.permute(0,2,1).contiguous()
        rays = input - view_point.unsqueeze(2) # [B,3, N]
        normalized_rays = rays/torch.norm(rays, dim=1, keepdim=True) # [B, 3, N]
        rays = self.rt(normalized_rays, rays, rays)
        d_view_point = self.vp2(rays.permute(0,2,1).contiguous(), False)
        adjusted_view_point = view_point+d_view_point
        adjusted_view_point = adjusted_view_point / torch.norm(adjusted_view_point, dim=1, keepdim=True)
        return view_point, adjusted_view_point


class Offset_Predictor(nn.Module):
    def __init__(self, dim_feat,offset_num, num_input_point=2048):
        super().__init__()
        self.offset_num = offset_num
        self.num_input_point = num_input_point
        self.mlp_Q = MLP_CONV(in_channel=3 + dim_feat, layer_dims=[256, 128])
        self.mlp_K = MLP_CONV(in_channel=3 + dim_feat, layer_dims=[256, 128])
        self.skip_transformer = SkipTransformer(in_channel=128, dim=64)
        self.ps = nn.ConvTranspose1d(dim_feat, 128, self.num_input_point, bias=True)
        self.mlp_dist = MLP_CONV(in_channel=128+3+3+128+512, layer_dims=[256,128,offset_num])

    def forward(self, global_feat, rays, input):
        ray_feat = torch.cat([global_feat.repeat(1,1,rays.shape[2]),
                            rays], 1)
        Q = self.mlp_Q(ray_feat)
        K = self.mlp_K(ray_feat)
        normalized_rays = rays/torch.norm(rays, dim=1, keepdim=True) # [B, 3, N]
        H = self.skip_transformer(normalized_rays, K, Q) # [B, 128, N] ray transformer
        P = self.ps(global_feat)
        H = torch.cat([H, P], dim=1)
        H_ = torch.cat([H, rays, input, global_feat.repeat(1,1,rays.shape[2])], dim=1)
        offset_dist = self.mlp_dist(H_)
        offset_dist = F.relu(offset_dist)
        return offset_dist, normalized_rays, H


class Offset_Adjustment(nn.Module):
    def __init__(self, offset_num):
        super().__init__()
        self.offset_num = offset_num
        self.mlp_dist = MLP_CONV(in_channel=256+512+512+3+3, layer_dims=[256,128,offset_num])
        self.PN_Encoder = PN_Encoder(512)
    
    def forward(self, H, coarse_pc, partial_global_feat, rays, input):
        """
        Args:
            H: (b, 256, 2048)
            coarse_pc: (b, 3, n)
            partial_global_feat: (b, 512, 1)
            rays: (b, 3, 2048)
            input: (b, 3, 2048)
        Returns:
            offset_dist: (b, offset_num, n)
        """
        coarse_global_feat = self.PN_Encoder(coarse_pc)
        feat = torch.cat([H, coarse_global_feat.repeat(1,1,rays.shape[2]), partial_global_feat.repeat(1,1,rays.shape[2]), rays, input], dim=1)
        offset_dist = self.mlp_dist(feat)
        offset_dist = torch.tanh(offset_dist)
        return offset_dist


class Offset_Model(nn.Module):
    def __init__(self, offset_num=2, dim_feat=512, num_input_point=2048):
        super().__init__()
        self.offset_num = offset_num
        self.num_input_point = num_input_point
        self.PN2_Encoder = PN2_Encoder(dim_feat)
        self.Offset_Predictor = Offset_Predictor(dim_feat, offset_num, num_input_point)
        self.Offset_Adjustment = Offset_Adjustment(offset_num)
    
    def forward(self, input, view_point, if_verbose=False):
        """
        Args:
            input: (b, 3, n)
            view_point: (b, 3)
        """
        pc_list = []
        off_dist_list = []
        
        input = input.permute(0,2,1).contiguous()
        rays = input - view_point.unsqueeze(2) # [B,3, N]
        partial_global_feat = self.PN2_Encoder(rays) # [B, dim_faet, 1]
        offset_dist1, normalized_rays, H = self.Offset_Predictor(partial_global_feat, rays, input)#[B, 8, N]
        off_dist_list.append(offset_dist1.permute(0,2,1).contiguous()) #[B, N, 8]
        
        pc0 = self.offset_along_rays(input, normalized_rays, offset_dist1)
        pc_list.append(pc0.permute(0,2,1).contiguous())
        
        offset_dist2 = self.Offset_Adjustment(H, pc0, partial_global_feat, rays, input)
        offset_dist2 = F.relu(offset_dist1+offset_dist2)
        off_dist_list.append(offset_dist2.permute(0,2,1).contiguous())

        pc1 = self.offset_along_rays(input, normalized_rays, offset_dist2)
        pc_list.append(pc1.permute(0,2,1).contiguous())
        if if_verbose:
            return pc_list[1], off_dist_list[1], partial_global_feat
        else:
            return pc_list, off_dist_list
    
    def offset_along_rays(self, input, normalized_rays, offset_dist):
        """
        Args:
            input: (b, 3, 2048)
            normalized_rays: (b, 3, 2048)
            offset_dist: (b, offset_num, 2048)
        Return:
            complete_pc: (b, 3, 2048*offset_num)
        """
        dist = offset_dist.view(-1,1,self.offset_num*self.num_input_point) # offset distance [B, 8, N]->[B, 1, 8XN]
        complete_pc = input.repeat(1,1,self.offset_num)+(normalized_rays.repeat(1,1,self.offset_num)).mul(dist) # [B, 3, 8XN]
        return complete_pc


class Offset_VP_Model(nn.Module):
    def __init__(self, offset_num=2, dim_feat=512, num_input_point=2048):
        super().__init__()
        self.offset_num = offset_num
        self.num_input_point = num_input_point
        self.PN2_Encoder = PN2_Encoder(dim_feat)
        self.Offset_Predictor = Offset_Predictor(dim_feat, offset_num, num_input_point)
        self.Offset_Adjustment = Offset_Adjustment(offset_num)
        self.ad_vp = Adjusted_VP_Model()
    
    def forward(self, input, if_verbose=False):
        """
        Args:
            input: (b, n, 3)
        returns:
            view_point: (b, 3)
        """
        pc_list = []
        off_dist_list = []
        _, view_point = self.ad_vp(input)
        input = input.permute(0,2,1).contiguous()
        rays = input - view_point.unsqueeze(2) # [B,3, N]
        partial_global_feat = self.PN2_Encoder(rays) # [B, dim_faet, 1]
        offset_dist1, normalized_rays, H = self.Offset_Predictor(partial_global_feat, rays, input)#[B, 8, N]
        off_dist_list.append(offset_dist1.permute(0,2,1).contiguous()) #[B, N, 8]
        
        pc0 = self.offset_along_rays(input, normalized_rays, offset_dist1)
        pc_list.append(pc0.permute(0,2,1).contiguous())
        
        offset_dist2 = self.Offset_Adjustment(H, pc0, partial_global_feat, rays, input)
        offset_dist2 = F.relu(offset_dist1+offset_dist2)
        off_dist_list.append(offset_dist2.permute(0,2,1).contiguous())

        pc1 = self.offset_along_rays(input, normalized_rays, offset_dist2)
        pc_list.append(pc1.permute(0,2,1).contiguous())
        if if_verbose:
            return pc_list[1], off_dist_list[1], partial_global_feat
        else:
            return pc_list, off_dist_list
    
    def offset_along_rays(self, input, normalized_rays, offset_dist):
        """
        Args:
            input: (b, 3, 2048)
            normalized_rays: (b, 3, 2048)
            offset_dist: (b, offset_num, 2048)
        Return:
            complete_pc: (b, 3, 2048*offset_num)
        """
        dist = offset_dist.view(-1,1,self.offset_num*self.num_input_point) # offset distance [B, 8, N]->[B, 1, 8XN]
        complete_pc = input.repeat(1,1,self.offset_num)+(normalized_rays.repeat(1,1,self.offset_num)).mul(dist) # [B, 3, 8XN]
        return complete_pc


class OCRU(nn.Module):
    def __init__(self, dim_feat=512, up_factor=2, i=0, radius=1):
        super(OCRU, self).__init__()
        self.i = i
        self.up_factor = up_factor
        self.radius = radius
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.mlp_2 = MLP_CONV(in_channel=128 * 2 + dim_feat, layer_dims=[256, 128])

        self.skip_transformer = SkipTransformer(in_channel=128, dim=64)

        self.mlp_ps = MLP_CONV(in_channel=128, layer_dims=[64, 32])
        self.ps = nn.ConvTranspose1d(32, 128, up_factor, up_factor, bias=False)   # point-wise splitting

        self.up_sampler = nn.Upsample(scale_factor=up_factor)
        self.mlp_delta_feature = MLP_Res(in_dim=256, hidden_dim=128, out_dim=128)

        self.mlp_delta = MLP_CONV(in_channel=128, layer_dims=[64, 3])

    def forward(self, pcd_prev, feat_global, offset_range, K_prev=None):
        """
        Args:
            pcd_prev: Tensor, (B, 3, N_prev)
            feat_global: Tensor, (B, dim_feat, 1)
            K_prev: Tensor, (B, 128, N_prev)

        Returns:
            pcd_child: Tensor, up sampled point cloud, (B, 3, N_prev * up_factor)
            K_curr: Tensor, displacement feature of current step, (B, 128, N_prev * up_factor)
        """
        b, _, n_prev = pcd_prev.shape
        feat_1 = self.mlp_1(pcd_prev)
        feat_1 = torch.cat([feat_1,
                            torch.max(feat_1, 2, keepdim=True)[0].repeat((1, 1, feat_1.size(2))),
                            feat_global.repeat(1, 1, feat_1.size(2))], 1)
        Q = self.mlp_2(feat_1)

        H = self.skip_transformer(pcd_prev, K_prev if K_prev is not None else Q, Q)

        feat_child = self.mlp_ps(H)
        feat_child = self.ps(feat_child)  # (B, 128, N_prev * up_factor)
        H_up = self.up_sampler(H)
        K_curr = self.mlp_delta_feature(torch.cat([feat_child, H_up], 1))
        offset_range = self.up_sampler(offset_range)

        delta = torch.tanh(self.mlp_delta(torch.relu(K_curr))) / (self.radius**self.i) # (B, 3, N_prev * up_factor)
        delta = delta * offset_range
        pcd_child = self.up_sampler(pcd_prev)
        pcd_child = pcd_child + delta

        return pcd_child, offset_range, K_curr


class SCPS_Model(nn.Module):
    def __init__(self, up_factors=None, offset_num=4, dim_feat=512, radius=1.5):
        super().__init__()
        self.offset_num = offset_num
        self.offset_model = Offset_Model(offset_num=offset_num, dim_feat=512, num_input_point=2048)

        if up_factors is None:
            up_factors = [1]
        else:
            up_factors = [1] + up_factors

        uppers = []
        for i, factor in enumerate(up_factors):
            uppers.append(OCRU(dim_feat=dim_feat, up_factor=factor, i=i, radius=radius))

        self.uppers = nn.ModuleList(uppers)

    def forward(self, input, view_point, if_off=False):
        """
        Args:
            input: (b, n, 3)
            view_point: (b, 3)
        Returns:
            arr_pcd: (b, N, 3) tensor list
        """
        arr_pcd = []
        coarse_pc, off_dist, partial_global_feat = self.offset_model(input, view_point, if_verbose=True)
        offset_dist = off_dist
        arr_pcd.append(coarse_pc)
        K_prev = None
        off_dist = off_dist.permute(0,2,1).contiguous().view(-1, 1, 2048*self.offset_num).permute(0,2,1).contiguous()
        
        pc, off_dist = fps_subsample_with_dist(coarse_pc, off_dist, 2048)
        pc = pc.permute(0, 2, 1).contiguous()
        off_dist = off_dist.permute(0, 2, 1).contiguous()
        offset_range = off_dist*(1.0 / len(self.uppers))+0.03

        for ocru in self.uppers:
            pc, offset_range, K_prev = ocru(pc, partial_global_feat, offset_range, K_prev)
            arr_pcd.append(pc.permute(0, 2, 1).contiguous())
        if if_off:
            return arr_pcd, offset_dist
        else:
            return arr_pcd


class SCPS_VP_Model(nn.Module):
    def __init__(self, up_factors=None, dim_feat=512, radius=1.5):
        super().__init__()
        self.offset_model = Offset_VP_Model(offset_num=4, dim_feat=512, num_input_point=2048)

        if up_factors is None:
            up_factors = [1]
        else:
            up_factors = [1] + up_factors

        uppers = []
        for i, factor in enumerate(up_factors):
            uppers.append(OCRU(dim_feat=dim_feat, up_factor=factor, i=i, radius=radius))

        self.uppers = nn.ModuleList(uppers)

    def forward(self, input):
        """
        Args:
            input: (b, n, 3)
        Returns:
            arr_pcd: (b, N, 3) tensor list
        """
        arr_pcd = []
        coarse_pc, off_dist, partial_global_feat = self.offset_model(input, if_verbose=True)
        arr_pcd.append(coarse_pc)
        K_prev = None
        off_dist = off_dist.permute(0,2,1).contiguous().view(-1, 1, 2048*4).permute(0,2,1).contiguous()
        
        pc, off_dist = fps_subsample_with_dist(coarse_pc, off_dist, 2048)
        pc = pc.permute(0, 2, 1).contiguous()
        off_dist = off_dist.permute(0, 2, 1).contiguous()
        offset_range = off_dist*(1.0 / len(self.uppers))+0.03

        for ocru in self.uppers:
            pc, offset_range, K_prev = ocru(pc, partial_global_feat, offset_range, K_prev)
            arr_pcd.append(pc.permute(0, 2, 1).contiguous())

        return arr_pcd



if __name__ == '__main__':
    pass

    # out = torch.randn(1,3,4)
    # print(out)
    # out = out.repeat(1,1,8).permute(0,2,1)
    # print(out)
    
    