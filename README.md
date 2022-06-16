# NVVC-RECT: Referring Expression Comprehension Transformer using Nonverbal and Verbal Cues
Code for CoRL 2022 submission Understanding Embodied Reference with Touch-Line Transformer. \
Modified from MDETR (https://github.com/ashkamath/mdetr) and YouRefIt (https://yixchen.github.io/YouRefIt). 

## Project Structure
    Project_NAME/
        ├── process_masks_and_images_for_MAT.ipynb/
        ├── main_ref.py/
        ├── pretrained/
        │   ├── 20_query_model.pth/
        │   ├── best_etf.pth/
        │   ├── best_arm.pth/
        │   ├── best_np.pth/
        │   └── best_ip.pth/
        ├── predictions/
        │   ├── arm.csv/
        │   ├── eye-to-fingertip.csv/
        │   ├── inpaint.csv/
        │   └── no_pose.csv/
        └── yourefit
            ├── images/
            ├── pickle/
            ├── paf/
            ├── saliency/
            ├── inpaint_Place_using_expanded_masks/
            ├── eye_to_fingertip/
            │   ├── eye_to_fingertip_annotations_train.csv/
            │   ├── eye_to_fingertip_annotations_valid.csv/
            │   ├── train_names.txt/
            │   └── valid_names.txt.txt/
            └── arm/

pretrained: a directory contains checkpoints.

pretrained/20_query_model.pth: we sliced (from 100 queries to 20 queries) the 
checkpoint of the checkpoint provided by the authors of [MDETR](https://github.com/ashkamath/mdetr) 

yourefit: a directory that contains the downloaded YouRefIt dataset. 
This directory will also contain inpainitings produced by readers. 
(Refer to the "inpainting" section for how to produce inpaintings). 

yourefit/eye_to_fingertip: a directory containing annotations for eyes and 
fingertips

yourefit/arm: annotations for arms.

## File that Pertains to the Scientific Claims the Most
models/mdetr.py

## Environment and Data

### 1. install dependencies
```bash
conda create --name nvvc python=3.8
conda activate nvvc
pip install -r requirements.txt
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```


### 2. download data
1. Download YouRefIt images and annotations as yourefit.zip
2. unzip yourefit.zip outside of this project and get a folder named "yourefit"
3. move or copy "images", "pickle", "paf", and "saliency" in the "yourefit" outside of this project into the existing "yourefit" folder inside this project

### 3. download checkpoints (pre-trained models)
Use hyperlinks in checkpoint column of the table below, and put them into the directory named "pretrained", which is under project root (refer to the "project structure" section above)

| Model               | precision: IoU=0.25 | precision: IoU=0.50 | precision: IoU=0.75 | checkpoint                                                                               |
|---------------------|---------------------|---------------------|---------------------|------------------------------------------------------------------------------------------|
| eye + fingertip     | 0.7002398081534772  | 0.6250999200639489  | 0.3820943245403677  | [best_etf.pth](https://www.icloud.com.cn/iclouddrive/0c06lZqRZijvwT4WqcMgBtiRw#best_etf) |
| elbow joint + wrist | 0.6786570743405276  | 0.5971223021582733  | 0.34772182254196643 | [best_arm.pth](https://www.icloud.com.cn/iclouddrive/012gtCECZ_TRZXkVNMZYlMVkA#best_arm) |
| no explicit pose    | 0.6370903277378097  | 0.5651478816946442  | 0.36211031175059955 | [best_np.pth](https://www.icloud.com.cn/iclouddrive/02c7z2s0sD7PMmSTukDiHNd4g#best_np)   |
| inpainting          | 0.5787370103916867  | 0.5091926458832934  | 0.31414868105515587 | [best_ip.pth](https://www.icloud.com.cn/iclouddrive/0cbyKDSeOP0gi1oBojieeoiJg#best_ip)   |
| MDETR               | -                   | -                   | -                   | [20_query_model.pth](https://www.icloud.com.cn/iclouddrive/007HLPMZ7qRE3ZudW9-cGUI8Q#20_query_model)                                                               |



### 4. (optional) generate inpaintings
We provide jupyter notebooks to expand the human masks required for inpainting,
Readers need to generate humans masks by themselves using F-RCNN because the 
yourefit dataset does not include human masks. The mask generation process
is straightforward. Readers can refer to the github repo created by the authors
of F-RCNNs for how to generate human masks. We only provide notebooks 
to expand and resize masks. Download by clicking the hyperlink.\
[process_masks_and_images_for_MAT.ipynb](https://www.icloud.com.cn/iclouddrive/097apIZkEWV4t6IDgv2ihPnqQ#process_masks_and_images_for_MAT)

After generating masks using the notebook, readers may, or may not, need to 
flip the values (e.g. change 255 to 0 and 0 to 255) in the output masks, 
depending on how readers generated humans masks using F-RCNN. After that, feed 
the masks and images to the model MAT for inpainting.\

After inpainting, readers may need to resize the inpaintings back to the 
sizes of the original images because the input and output of MAT are squares.
If readers reshaped expanded masks to square (instead of masking them) before 
feedings them into MAT, readers need to reshape the MAT output back to 
original sizes. In contrast, if readers choose to mask, readers can process the
MAT outputs by cropping them. We only provide the notebook to reshape square 
outputs back to the sizes of original images. \
[restore_inpaint_size.ipynb](https://www.icloud.com.cn/iclouddrive/0b4JVNYx6bpT542w59-YnZiVw#restore_inpaint_size) \
Note that readers need to modify the image_dir, inpaint_dir, and output_dir in 
the notebook provided above. 
(image_dir is the path to yourefit images. the shape of the original images 
will be used. inpaint dir is the path to the MAT outputs. output_dir 
is the path to store the inpainted images that are reshaped to the sizes of 
original images by the above notebook)

Finally, after obtaining inpaintings, change the INPAINT_DIR in magic_numbers.py
to the path of the inpainted images that are reshaped to sizes of origianl
images. Note that INPAINT_DIR is a relative path 
(relative to Project_NAME/yourefit. Please refer to the project 
structure section). 


## Evaluate
### eye + fingertip
use the unmodified magic_numbers.py and run:
```bash
python main_ref.py --num_workers=1 --dataset_config configs/yourefit.json --batch_size 1   --ema --text_encoder_lr 1e-4 --lr 5e-5 --output-dir 'output_dir/eval' --resume pretrained/best_etf.pth --eval
```

### elbow joint + wrist
before running, in unmodified magic_numbers.py, set:\
REPLACE_ARM_WITH_EYE_TO_FINGERTIP = False
```bash
python main_ref.py --num_workers=1 --dataset_config configs/yourefit.json --batch_size 1   --ema --text_encoder_lr 1e-4 --lr 5e-5 --output-dir 'output_dir/eval' --resume pretrained/best_arm.pth --eval
```

### no explicit pose 
before running, in unmodified magic_numbers.py, set:\
RESERVE_QUERIES_FOR_ARMS = False\
NUM_RESERVED_QUERIES_FOR_ARMS = 0

```bash
python main_ref.py --num_workers=1 --dataset_config configs/yourefit.json --batch_size 1   --ema --text_encoder_lr 1e-4 --lr 5e-5 --output-dir 'output_dir/eval' --resume pretrained/best_np.pth --eval --pose False
```

### inpainting (requires inpaintings)
(requires generated inpintings, see the optional "generate inpaintings" section in the "reproduction" section)
before running, in unmodified magic_numbers.py, set:\
RESERVE_QUERIES_FOR_ARMS = False\
NUM_RESERVED_QUERIES_FOR_ARMS = 0\
REPLACE_IMAGES_WITH_INPAINT = True
```bash
python main_ref.py --num_workers=1 --dataset_config configs/yourefit.json --batch_size 1   --ema --text_encoder_lr 1e-4 --lr 5e-5 --output-dir 'output_dir/eval' --resume pretrained/best_ip.pth --eval --pose False
```

## Train
### eye + fingertip
use the unmodified magic_numbers.py and run:
```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port 64331 --use_env main_ref.py --num_workers 8 --dataset_config configs/yourefit.json --batch_size 7   --ema --text_encoder_lr 1e-4 --lr 5e-5 --output-dir 'output_dir/debug_etf' --load pretrained/20_query_model.pth
```

### elbow joint + wrist
before running, in unmodified magic_numbers.py, set:\
REPLACE_ARM_WITH_EYE_TO_FINGERTIP = False
```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port 64332 --use_env main_ref.py --num_workers 8 --dataset_config configs/yourefit.json --batch_size 7   --ema --text_encoder_lr 1e-4 --lr 5e-5 --output-dir 'output_dir/debug_arm' --load pretrained/20_query_model.pth
```

### no explicit pose
before running, in unmodified magic_numbers.py, set:\
RESERVE_QUERIES_FOR_ARMS = False\
NUM_RESERVED_QUERIES_FOR_ARMS = 0
```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port 64333 --use_env main_ref.py --num_workers 8 --dataset_config configs/yourefit.json --batch_size 7   --ema --text_encoder_lr 1e-4 --lr 5e-5 --output-dir 'output_dir/debug_np' --load pretrained/20_query_model.pth --pose False
```

### inpainting
(requires generated inpintings, see the optional "generate inpaintings" section in the "reproduction" section)
before running, in unmodified magic_numbers.py, set:\
RESERVE_QUERIES_FOR_ARMS = False\
NUM_RESERVED_QUERIES_FOR_ARMS = 0\
REPLACE_IMAGES_WITH_INPAINT = True
```bash
python -m torch.distributed.launch --nproc_per_node=8 --master_port 64334 --use_env main_ref.py --num_workers 8 --dataset_config configs/yourefit.json --batch_size 7   --ema --text_encoder_lr 1e-4 --lr 5e-5 --output-dir 'output_dir/debug_ip' --load pretrained/20_query_model.pth --pose False
```


## Visualizations
We provide jupyter notebooks to visualize predictions stored in csv files, which can be obtained by:\
setting SAVE_EVALUATION_PREDICTIONS = True and run any of the evaluation command provided in the evaluation section above.

[cleaned_visualize_predictions_eye_to_fingertip.ipynb](https://www.icloud.com.cn/iclouddrive/085wsL7MVOS36Gj2FnbCFO0cw#cleaned_visualize_predictions_eye_to_fingertip) \
[cleaned_visualize_predictions_elbow_joint_to_wrist.ipynb](https://www.icloud.com.cn/iclouddrive/0552kQOlE-QikukvaIkIKm-jw#cleaned_visualize_predictions_elbow_joint_to_wrist) \
[cleaned_visualize_predictions_no-pose.ipynb](https://www.icloud.com.cn/iclouddrive/0a3vivLQiLPZsQtDJgzebWbqA#cleaned_visualize_predictions_no-pose) \
[cleaned_visualize_predictions_inpaint.ipynb](https://www.icloud.com.cn/iclouddrive/07bTPib5HN7beipUBb6VgaM8A#cleaned_visualize_predictions_inpaint)

