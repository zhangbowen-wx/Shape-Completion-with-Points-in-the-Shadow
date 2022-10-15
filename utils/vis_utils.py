import numpy as np
import torch


class ColorPointCloud:
    def __init__(self, xyz, offset_num):
        self.xyz = xyz # xyz coordinates of point cloud [N,3]
        self.pointcloud_num = xyz.shape[0]
        self.offset_num = offset_num
        self.layer_mapping = {'0': 'white',
                              '1': 'red',
                              '2': 'orange',
                              '3': 'yellow',
                              '4': 'green',
                              '5': 'cyan',
                              '6': 'blue',
                              '7': 'purple'}
        self.color_mapping = {'white': np.array([[255,255,255]]),
                              'red': np.array([[255,0,0]]),
                              'orange': np.array([[255,165,0]]),
                              'yellow': np.array([[255,255,0]]),
                              'green': np.array([[0,255,0]]),
                              'cyan': np.array([[0,127,255]]),
                              'blue': np.array([[0,0,255]]),
                              'purple': np.array([[139,0,255]])}
        self.color_pc = [] # [N, 3+3]
    
    def save_ply(self, path):
        f = open(path, 'w')
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment PCL generated\n")
        f.write("element vertex %d\n"%self.pointcloud_num)
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(self.pointcloud_num):
            f.write("%.5f %.5f %.5f %d %d %d\n"%(self.color_pc[i,0],self.color_pc[i,1],self.color_pc[i,2],self.color_pc[i,3],self.color_pc[i,4],self.color_pc[i,5]))
        f.close()

    @classmethod
    def save_offset_color(cls, complete_pc, offset_num, path):
        pc = cls(complete_pc, offset_num)
        pc_list = []
        for i in range(offset_num):
            pc_list.append(pc.xyz[i*(pc.pointcloud_num//pc.offset_num):(i+1)*(pc.pointcloud_num//pc.offset_num), :])
        for i in range(len(pc_list)):
            colors = np.tile(pc.color_mapping[pc.layer_mapping[str(i)]], (pc.pointcloud_num//pc.offset_num, 1))
            sub_color_pc = np.concatenate([pc_list[i], colors], axis=1)
            pc.color_pc.append(sub_color_pc)
        pc.color_pc = np.concatenate(pc.color_pc, axis=0)
        pc.save_ply(path)
    
    @classmethod
    def save_pure_color(cls, pc, path, *, pure_color='yellow'):
        pc = cls(pc, None)
        if pure_color not in pc.color_mapping.keys():
            raise RuntimeError("%s RGB calue is not included in color mapping."%(pure_color))
        pc.color_pc = np.concatenate([pc.xyz, np.tile(pc.color_mapping[pure_color], (pc.pointcloud_num, 1))], axis=1)
        pc.save_ply(path)


if __name__ == '__main__':
    # N = 2
    # view_point = torch.tensor([[0,0,0]]).float().unsqueeze(2) # [1, 3]
    # # partial_pc = torch.rand((1, 3, N)) # [1, 3, N]
    # partial_pc = torch.tensor([[[1,0],
    #                             [0,1],
    #                             [0,0]]]).float()
    # rays = partial_pc-view_point
    # normalized_rays = rays/torch.norm(rays, dim=1, keepdim=True)# [1, 3, N]
    # print(normalized_rays)
    # normalized_rays = normalized_rays.repeat(1,1,8)
    # print(normalized_rays)
    # # x = torch.rand((1, 1, N))
    # x = torch.tensor([[[1, 10]]]).float()
    # x = torch.cat([x*0, x*1, x*2, x*3, x*4, x*5, x*6, x*7], dim=1)# [1,8,N]
    # print(x)
    # dist = x.view(-1, 1, 8*N).repeat(1,3,1)#[1,3,N*8]
    # print(dist)
    # print(partial_pc)
    # print(partial_pc.repeat(1,1,8))

    # complete_pc = partial_pc.repeat(1,1,8)+normalized_rays.mul(dist)
    # complete_pc = complete_pc.permute(0,2,1).view(-1,3).numpy()
    # ColorPointCloud.save_offset_color(complete_pc, 8, 'x.ply')
    # np.savetxt('x.xyz', partial_pc.permute(0,2,1).view(N,3).numpy())


    partial_point_cloud = torch.rand((2048,3)).numpy()
    ColorPointCloud.save_pure_color(partial_point_cloud, 'test.ply')













