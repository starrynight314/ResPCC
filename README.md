# README
## Install packages
Before running the code, please ensure the environment is configured with the following dependencies:

```
Python 3.7
PyTorch 1.7.1 (CUDA 11.0)
NumPy 1.20.3
SciPy 1.7.3
scikit-learn 1.0.1
Open3D 0.9.0.0
einops 0.3.2
tqdm
pandas 1.3.5
matplotlib 3.5.3
plyfile 0.9
compressai 1.2.4
pytorch-msssim 1.0.0
Chamfer3D for calculating Chamfer distance
pointops
Manifold for preparing ShapeNet dataset
```

## Directory Structure

```
├── args/                   # Configuration files            
├── dataset/                # Dataset classes with getitem
├── exp/                    # Test GE sequences for different packet loss rates
├── log/                    # Create the folder to store logs generated during training
├── metrics/                # Compute D1 and D2 PSNR metrics
├── models/                 # Neural network architectures
├── output/                 # Create the folder to save checkpoints during training
├── results/                # Create the folder to store reconstructed results during testing
├── hw_lookup.pkl           # Lookup table reshaped from 1D to 2D
├── look_up_table.py        # Function to generate the lookup table   
├── prepare_semantickitti.py       # Process SemanticKITTI dataset into patches 
├── prepare_shapenet.py     # Process ShapeNet dataset 
├── README.md
├── test.py                 # Testing script
├── train_refine.py         # Fine-tuning script: freezes backbone, updates dictionary and decoder modules
└── train.py                # Main training script
```

## Data Preparation

Download ShapeNetCore (https://huggingface.co/datasets/ShapeNet/ShapeNetCore-archive) and SemanticKITTI (http://semantic-kitti.org/dataset.html#download) datasets

```
python prepare_shapenet.py
python prepare_semantickitti.py
```

Make sure to set `--data_root` to the correct path of your raw dataset.

## Training

```
python train.py (w/o refinement_net) for 50 epoches
then 

python train_refine.py (w/ refinement_net) for 15 epoches
```

Make sure to set `--train_data_path` `--valid_data_path` to the correct path

## Testing

```
python test.py 
```

Make sure to set `--test_data_path` to the correct path and also set `--model_path` accordingly.

 

