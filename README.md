# Melt electrowriting enabled 3D liquid crystal elastomer structures for cross-scale actuators and temperature field sensors 
This is the official implementation of [Melt electrowriting enabled 3D liquid crystal elastomer structures for cross-scale actuators and temperature field sensors ]

## Installation

1) Clone this repository

2) create a virtual environment in CONDA and enter: 
```
conda create -n Melt-EHD python=3.8
conda activate Melt-EHD
```

3) If you don't need to process your data set, you can skip this step:```pip install pyyaml opencv-python pandas mvtec-halcon==21050.0.0```
(If you don't have halcon, you need to install halcon first.)

4) ```pip install numpy pandas matplotlib scipy tensorboard```

5) ```conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge torchkeras tqdm accelerate```

## Dataset 

We have provided the processed data set in dataset/trian and dataset/test, and you can use them directly. And you can also use following steps to process your own data set.

1) extract_data folder to create a new [number] folder as the current working condition of the main folder (mainDir), a new csv folder to save the next line of csv file (such as E:\t_map\extract_data\1\csv)

2) Extract video images. 
Modify VIDEO_PATH and mainDir in dataset/preprocess/config.yaml.
Switch to dataset/preprocess/ and type: ```python 1extract_image_data.py```

3) Extract the corner points. 
Modify the cornerPath and alignTxtPath in config.yaml.
```python 2get_corner.py```
Take the four sharp edges on the frame in order [top left, top right, bottom left, bottom right].

4) Time alignment.
```python 3registerTansform.py```
Input to help distinguish aligned temperature chart numbers and find the best blend chart numbers.

5) Extract the csv file for making the data set from the thermal imager exe data and put it into the csv folder

6) Start labeling.
Modify the x in config.yaml in the pre_data folder.
```python 4createDataset.py```

  6.1 Draw the polygon surrounding box, surround the 14 lines in the middle, left click to add a point, right click to delete the point, enter to close the polygon to complete the surrounding.

  6.2 Follow the prompts to check points.

  6.3 Follow the prompts to check numbers.

7) You can check the data.
```python 5checkDataset.py```

8) Divide the data set into a training set and a test set.
```python ../../datasetDivide.py```
(You need to change the folder_list and target_folder in datasetDivide.py to yourself.)

## Training

```
python AI_model.py --train True --batch_size 128 --dataset_dir /path_to_dataset --model_dir /where_to_save_model
```

## Evaluating

```
python AI_model.py --train False --load_model_path /where_to_load_model --result_dir /where_to_save_results
```

## Citation

Please cite [Melt electrowriting enabled 3D liquid crystal elastomer structures for cross-scale actuators and temperature field sensors](https://github.com/XueZhengjie/Melt-EHD/) if you use it in your research


## License

Our codes are licensed under the Creative Commons Attribution-NonCommercial 4.0 International license and is freely available for non-commercial use. Please see the LICENSE for further details. If you are interested in commercial use, please contact us under fengxueming@stu.xjtu.edu.cn or xuezhengjie@stu.xjtu.edu.cn.
