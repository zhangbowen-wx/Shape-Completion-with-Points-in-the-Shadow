import os
import h5py
import numpy as np
import open3d as o3d


category_name_list = [
        "airplane", "cabinet", "car", "chair",
        "lamp", "sofa", "table", "watercraft",
        "bed", "bench", "bookself", "bus",
        "guitar", "motorbike", "pistol", "skateboard"
        ]


if __name__ == '__main__':
    data_dir = "./data"
    save_dir = "./datasets/VMVP"

    ##################################################
    # make input h5 file
    ##################################################
    for state in ['train', 'test']:
        labels = []
        data = []
        view_points = []
        f = h5py.File(os.path.join(save_dir, 'vmvp_%s_input.h5'%(state)), 'w')
        c = 0
        print('-'*30)
        for category_name in category_name_list:
            list_file = open('./train_test_lists/%s_%s.txt'%(category_name, state), 'r')
            lines = list_file.readlines()
            list_file.close()
            print(len(lines))
            for line in lines:
                line_split = line.strip('\n').split(' ')
                category_id = line_split[0]
                model_id = line_split[1]
                for i in range(26):
                    labels.append(c)
                    data.append(np.loadtxt(os.path.join("%s/%s/%s/partial_sampled/"%(data_dir, category_name, state), "%s_%s"%(category_id, model_id), "%d.xyz"%i)).astype(np.float32))
                    view_points.append(np.loadtxt(os.path.join("./render/%s/%s/view_point/"%(category_name, state), "%s_%s"%(category_id, model_id), "%d.xyz"%i)).astype(np.float32))
            c+=1
        data = np.array(data)
        labels = np.array(labels)
        view_points = np.array(view_points)
        f.create_dataset("labels", data=labels)
        f.create_dataset("incomplete_pcds", data=data)
        f.create_dataset("view_points", data=view_points)
        f.close()

    ##################################################
    # make gt point cloud h5 file
    ##################################################
    npts = 16384
    for state in ['train', 'test']:
        labels = []
        data = []
        f = h5py.File(os.path.join(save_dir, 'vmvp_%s_gt_%dpts.h5'%(state, npts)), 'w')
        c = 0
        for category_name in category_name_list:
            list_file = open('./train_test_lists/%s_%s.txt'%(category_name, state), 'r')
            lines = list_file.readlines()
            list_file.close()
            for line in lines:
                line_split = line.strip('\n').split(' ')
                category_id = line_split[0]
                model_id = line_split[1]
                labels.append(c)
                pcd = o3d.io.read_point_cloud(os.path.join("%s/%s/%s/complete/"%(data_dir, category_name, state), "%s_%s"%(category_id, model_id), "gt_%d.pcd"%npts))
                pcd = np.asarray(pcd.points)
                data.append(pcd)
            c+=1
        data = np.array(data)
        labels = np.array(labels)
        f.create_dataset("complete_pcds", data=data)
        f.create_dataset("labels", data=labels)
        f.close()
