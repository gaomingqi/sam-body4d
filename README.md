# SAM-4D-Body

Promptable Human Mesh Recovery from Videos, based on [SAM-3](https://github.com/facebookresearch/sam3) and [SAM-3D-Body](https://github.com/facebookresearch/sam-3d-body).

## 🖥️ Installation

#### 1. Create and Activate Environment
```
conda create -n 4db python=3.12 -y
conda activate 4db
```
#### 2. Install PyTorch (choose the version that matches your CUDA), Detectron, and SAM3
```
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps
pip install -e models/sam3
```
If you are using a different CUDA version, please select the matching PyTorch build from the official download page:
https://pytorch.org/get-started/previous-versions/

#### 3. Install Dependencies
```
pip install -e .
```

## 🚀 Run the Demo
#### 1. Download checkpoints of [SAM3](https://huggingface.co/facebook/sam3), [SAM-3D-Body](https://huggingface.co/facebook/sam-3d-body-dinov3), and [MoGe-2](https://huggingface.co/Ruicheng/moge-2-vitl-normal)

#### 2. Set checkpoint path in [`configs/4db.yaml`](configs/4db.yaml)

#### 3. Run:
```
python app.py
```

## 📖 Citation


## 👏 Acknowledgements

The project is built upon [SAM-3](https://github.com/facebookresearch/sam3) and [SAM-3D-Body](https://github.com/facebookresearch/sam-3d-body). We sincerely thank the original authors for their outstanding work and contributions. 
