# Data Generation 
If you want to generate data on your own, these python scripts maybe helpful. 

# Usage

## Step1: Scan CAD models obtain depth images and viewpoints.

Place shapenet data in the ./shapenet directory. 

To use Blender for depth image rendering, run following command.
```
blender -b -P render_depth.py
```
Generated raw data including depth images and viewpoints will be placed in ./render directory.

## Step2: Process depth images obtain partial point clouds.

To process EXR depth images, run following command:
```
python process_exr.py
```
Partial point clouds will be genearted in ./data directory.

## Step3: Sample partial point clouds.
Run following commands to sample partial point clouds.
```
cd ./sample
python generate_sampled_partial_input.py
```
Sampled partial point clouds will be placed in ./data directory.

## Step4: Sample on the CAD models to generate GT point clouds.
Run following command to generate GT point clouds.
```
python generate_gt.py
```
GT point clouds will be generated in ./data directory.

## Step5: Generate H5 files.
Run following commands to generate H5 files.
```
cd ../
python generate_hdf5_dataset.py
```
Generated H5 files will be placed in ./datasets/VMVP directory.








