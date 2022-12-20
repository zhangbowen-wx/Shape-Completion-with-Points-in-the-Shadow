import IFPS
import RS
import numpy as np
import torch
import os
import sys

category_name_list = [
        "airplane", "cabinet", "car", "chair",
        "lamp", "sofa", "table", "watercraft",
        "bed", "bench", "bookself", "bus",
        "guitar", "motorbike", "pistol", "skateboard"
        ]

sampled_num = 2048

pc_base_dir = '../data/'


for category_name in category_name_list:
    count = 0
    for state in ['train', 'test']:
        if os.path.exists(os.path.join(pc_base_dir, category_name, state, 'partial_sampled'))==False:
            os.mkdir(os.path.join(pc_base_dir, category_name, state, 'partial_sampled'))
        f = open(os.path.join('../train_test_lists', "%s_%s.txt"%(category_name, state)), 'r')
        lines = f.readlines()
        f.close()
        for line in lines:
            line_split = line.strip('\n').split(' ')
            category_id = line_split[0]
            model_id = line_split[1]
            count+=1
            sys.stdout.write("%s: %04d \r"%(category_name, count))
            sys.stdout.flush()
            if os.path.exists(os.path.join(pc_base_dir, category_name, state, 'partial_sampled', category_id+'_'+model_id))==False:
                os.mkdir(os.path.join(pc_base_dir, category_name, state, 'partial_sampled', category_id+'_'+model_id))
            for i in range(26):
                pc = np.loadtxt(os.path.join(pc_base_dir, category_name, state, 'partial', category_id+'_'+model_id, "%d.xyz"%(i))).astype(np.float32)
                if pc.shape[0]<=sampled_num:
                    sampled_pc = RS.resample_pcd(pc, sampled_num)
                    np.savetxt(os.path.join(pc_base_dir, category_name, state, 'partial_sampled', category_id+'_'+model_id, "%d.xyz"%(i)), sampled_pc)
                else:
                    pc = torch.from_numpy(pc)
                    pc = pc.view(1,-1,3)
                    sampled_pc_centroids = IFPS.farthest_point_sample(pc, sampled_num)
                    sampled_pc = pc[:,sampled_pc_centroids[0,:],:]
                    sampled_pc = sampled_pc.view(-1,3).numpy()
                    np.savetxt(os.path.join(pc_base_dir, category_name, state, 'partial_sampled', category_id+'_'+model_id, "%d.xyz"%(i)), sampled_pc)

