import bpy
import mathutils
import numpy as np
import os
import sys
import time

def random_rotation():
    angle_x = np.random.uniform() * 2 * np.pi
    angle_y = np.random.uniform() * 2 * np.pi
    angle_z = np.random.uniform() * 2 * np.pi
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R

def cal_pose(angle_x, angle_y, angle_z, radius, random_R):
    angle_x = angle_x / 180.0 * np.pi
    angle_y = angle_y / 180.0 * np.pi
    angle_z = angle_z / 180.0 * np.pi
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    R = np.dot(random_R, R)
    # Set camera pointing to the origin and 1 unit away from the origin
    t = np.expand_dims(R[:, 2]*radius, 1)
    pose = np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0)
    return pose


def setup_blender(width, height, focal_length):
    # camera
    camera = bpy.data.objects['Camera']
    camera.data.angle = np.arctan(width / 2 / focal_length) * 2

    # render layer
    scene = bpy.context.scene
    scene.render.filepath = 'buffer'
    scene.render.image_settings.color_depth = '16'
    scene.render.resolution_percentage = 100
    scene.render.resolution_x = width
    scene.render.resolution_y = height

    # compositor nodes
    scene.use_nodes = True
    tree = scene.node_tree
    rl = tree.nodes.new('CompositorNodeRLayers')
    output = tree.nodes.new('CompositorNodeOutputFile')
    output.base_path = ''
    output.format.file_format = 'OPEN_EXR'
    tree.links.new(rl.outputs['Depth'], output.inputs[0])

    # remove default cube
    # if bpy.context.object.mode == 'EDIT':
    #     bpy.ops.object.mode_set(mode='OBJECT')
    if 'Cube' in bpy.data.objects:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects['Cube'].select_set(True)
        bpy.ops.object.delete()

    return scene, camera, output

def cal_view_point(M):
    view_point = np.array([[M[0,3], M[1,3], M[2,3]]])
    return view_point

if __name__ == '__main__':
    data_dir = "./shapenet"
    save_dir = "./render"
    data_list_dir = "./train_test_lists"
    
    category_name_list = ["airplane", "cabinet", "car", "chair", 
                          "lamp", "sofa", "table", "watercraft", 
                          "bed", "bench", "bookself", "bus", 
                          "guitar", "motorbike", "pistol", "skateboard"
                          ]

    for category_name in category_name_list:
        for state in ["train", "test"]:
            output_dir = os.path.join(save_dir, category_name, state)
            
            width = 1600
            height = 1200
            focal = 200
            scene, camera, output = setup_blender(width, height, focal)
            intrinsics = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])

            open('blender.log', 'w+').close()
            if os.path.exists(output_dir)==False:
                os.makedirs(output_dir)
            np.savetxt(os.path.join(output_dir, 'intrinsics.txt'), intrinsics, '%f')

            f = open(os.path.join(data_list_dir, category_name+"_"+state+".txt"), 'r')
            lines = f.readlines()
            f.close()
            for line in lines:
                line_split = line.strip('\n').split(' ')
                category_id = line_split[0]
                model_id = line_split[1]
                start = time.time()
                exr_dir = os.path.join(output_dir, 'exr', category_id+"_"+model_id)
                pose_dir = os.path.join(output_dir, 'pose', category_id+"_"+model_id)
                view_point_dir = os.path.join(output_dir, 'view_point', category_id+"_"+model_id)
                if os.path.exists(exr_dir)==False:
                    os.makedirs(exr_dir)
                if os.path.exists(pose_dir)==False:
                    os.makedirs(pose_dir)
                if os.path.exists(view_point_dir)==False:
                    os.makedirs(view_point_dir)

                # Redirect output to log file
                old_os_out = os.dup(1)
                os.close(1)
                os.open('blender.log', os.O_WRONLY)

                # Import mesh model
                model_path = os.path.join(data_dir, category_id, model_id, 'model.obj')
                bpy.ops.import_scene.obj(filepath=model_path)

                # Rotate model by 90 degrees around x-axis (z-up => y-up) to match ShapeNet's coordinates
                bpy.ops.transform.rotate(value=-np.pi / 2, orient_axis='X')

                # Render
                random_R = random_rotation()
                i = 0
                for angle_x in [-45, 0, 45]:
                    for angle_y in range(0, 360, 45):
                        scene.frame_set(i)
                        pose = cal_pose(angle_x, angle_y, 0, 1, random_R)
                        camera.matrix_world = mathutils.Matrix(pose)
                        output.file_slots[0].path = os.path.join(exr_dir, '#.exr')
                        bpy.ops.render.render(write_still=True)
                        np.savetxt(os.path.join(pose_dir, '%d.txt' % i), pose, '%f')
                        np.savetxt(os.path.join(view_point_dir, '%d.xyz' % i), cal_view_point(pose), '%f')
                        i+=1
                scene.frame_set(i)
                pose = cal_pose(-90, 0, 0, 1, random_R)
                camera.matrix_world = mathutils.Matrix(pose)
                output.file_slots[0].path = os.path.join(exr_dir, '#.exr')
                bpy.ops.render.render(write_still=True)
                np.savetxt(os.path.join(pose_dir, '%d.txt' % i), pose, '%f')
                np.savetxt(os.path.join(view_point_dir, '%d.xyz' % i), cal_view_point(pose), '%f')
                i+=1
                scene.frame_set(i)
                pose = cal_pose(90, 0, 0, 1, random_R)
                camera.matrix_world = mathutils.Matrix(pose)
                output.file_slots[0].path = os.path.join(exr_dir, '#.exr')
                bpy.ops.render.render(write_still=True)
                np.savetxt(os.path.join(pose_dir, '%d.txt' % i), pose, '%f')
                np.savetxt(os.path.join(view_point_dir, '%d.xyz' % i), cal_view_point(pose), '%f')
                # Clean up
                bpy.ops.object.delete()
                for m in bpy.data.meshes:
                    bpy.data.meshes.remove(m)
                for m in bpy.data.materials:
                    m.user_clear()
                    bpy.data.materials.remove(m)

                # Show time
                os.close(1)
                os.dup(old_os_out)
                os.close(old_os_out)
                print('%s done, time=%.4f sec' % (model_id, time.time() - start))
                print(model_path)
