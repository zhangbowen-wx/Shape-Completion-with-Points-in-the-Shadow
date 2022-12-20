import open3d as o3d
import numpy as np
import os
import sys

category_name_list = [
        "airplane", "cabinet", "car", "chair",
        "lamp", "sofa", "table", "watercraft",
        "bed", "bench", "bookself", "bus",
        "guitar", "motorbike", "pistol", "skateboard"
        ]

pc_base_dir = '../data'
shapenet_dir = '../shapenet'

for sampled_num in [2048, 4096, 8192, 16384]:
    for category_name in category_name_list:
        count = 0
        for state in ['train', 'test']:
            if os.path.exists(os.path.join(pc_base_dir, category_name, state, 'complete'))==False:
                os.mkdir(os.path.join(pc_base_dir, category_name, state, 'complete'))
            f = open('../render/train_test_lists/%s_%s.txt'%(category_name, state), 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                count+=1
                sys.stdout.write("%s: %04d \r"%(category_name, count))
                sys.stdout.flush()
                line_split = line.strip('\n').split(' ')
                category_id = line_split[0]
                model_id = line_split[1]
                model_path = '%s/%s/%s/model.obj'%(shapenet_dir, category_id, model_id)
                mtl_path = '%s/%s/%s/model.mtl'%(shapenet_dir, category_id, model_id)
                os.system('rm %s'%(mtl_path))
                mesh = o3d.io.read_triangle_mesh(model_path)
                pcd = mesh.sample_points_poisson_disk(sampled_num)
                if os.path.exists(os.path.join(pc_base_dir, category_name, state, 'complete', '%s_%s'%(category_id, model_id)))==False:
                    os.mkdir(os.path.join(pc_base_dir, category_name, state, 'complete', '%s_%s'%(category_id, model_id)))
                save_path = os.path.join(pc_base_dir, category_name, state, 'complete', '%s_%s'%(category_id, model_id), 'gt_%d.pcd'%(sampled_num))
                o3d.io.write_point_cloud(save_path, pcd)

