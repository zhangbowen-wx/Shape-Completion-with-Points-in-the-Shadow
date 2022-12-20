'''
MIT License

Copyright (c) 2018 Wentao Yuan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import Imath
import OpenEXR
import argparse
import array
import numpy as np
import os
import sys


def read_exr(exr_path, height, width):
    file = OpenEXR.InputFile(exr_path)
    depth_arr = array.array('f', file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT)))
    depth = np.array(depth_arr).reshape((height, width))
    depth[depth < 0] = 0
    depth[np.isinf(depth)] = 0
    return depth


def depth2pcd(depth, intrinsics, pose):
    inv_K = np.linalg.inv(intrinsics)
    inv_K[2, 2] = -1
    depth = np.flipud(depth)
    y, x = np.where((depth > 0) & (depth<5))
    # image coordinates -> camera coordinates
    points = np.dot(inv_K, np.stack([x, y, np.ones_like(x)] * depth[y, x], 0))
    # camera coordinates -> world coordinates
    points = np.dot(pose, np.concatenate([points, np.ones((1, points.shape[1]))], 0)).T[:, :3]
    return points


if __name__ == '__main__':
    render_base_dir = './render' 
    pcd_base_dir = './data'
    category_name_list = [
        "airplane", "cabinet", "car", "chair",
        "lamp", "sofa", "table", "watercraft",
        "bed", "bench", "bookself", "bus",
        "guitar", "motorbike", "pistol", "skateboard"
        ]
    # parser = argparse.ArgumentParser()
    for category_name in category_name_list:
        count = 0
        for state in ['train', 'test']:
            intrinsics_file = os.path.join(render_base_dir, category_name, state, 'intrinsics.txt')
            num_scans = 26

            intrinsics = np.loadtxt(intrinsics_file)
            width = int(intrinsics[0, 2] * 2)
            height = int(intrinsics[1, 2] * 2)
            f = open(os.path.join('train_test_lists', "%s_%s.txt"%(category_name, state)), 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                line_split = line.strip('\n').split(' ')
                category_id = line_split[0]
                model_id = line_split[1]
                count+=1
                sys.stdout.write("%s: %04d \r"%(category_name, count))
                sys.stdout.flush()
                pcd_dir = os.path.join(pcd_base_dir, category_name, state, 'partial', category_id+'_'+model_id)
                os.makedirs(pcd_dir, exist_ok=True)
                for i in range(num_scans):
                    exr_path = os.path.join(render_base_dir, category_name, state, 'exr', category_id+'_'+model_id, '%d.exr' % i)
                    pose_path = os.path.join(render_base_dir, category_name, state, 'pose', category_id+'_'+model_id, '%d.txt' % i)
                    depth = read_exr(exr_path, height, width)
                    pose = np.loadtxt(pose_path)
                    points = depth2pcd(depth, intrinsics, pose)
                    
                    np.savetxt(os.path.join(pcd_dir, '%d.xyz' % i), points)
