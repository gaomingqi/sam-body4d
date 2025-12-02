# Using Diffusion Priors for Video Amodal Segmentation

**CVPR 2025**

Official implementation of <strong>Using Diffusion Priors for Video Amodal Segmentation</strong>

[*Kaihua Chen*](https://www.linkedin.com/in/kaihuac/), [*Deva Ramanan*](https://www.cs.cmu.edu/~deva/), [*Tarasha Khurana*](https://www.cs.cmu.edu/~tkhurana/)

![diffusion-vas](assets/diffusion-vas.gif)

[**Paper**](https://arxiv.org/abs/2412.04623) | [**Project Page**](https://diffusion-vas.github.io)

## TODO 🤓

- [x] Release the checkpoint and inference code 
- [x] Release evaluation code for SAIL-VOS and TAO-Amodal
- [x] Release fine-tuning code for Diffusion-VAS

## Getting Started

### Installation

#### 1. Clone the repository

```bash
git clone https://github.com/Kaihua-Chen/diffusion-vas
cd diffusion-vas
```

#### 2. Create and activate a virtual environment

```bash
conda create --name diffusion_vas python=3.10
conda activate diffusion_vas
pip install -r requirements.txt
```

### Download Checkpoints

We provide our Diffusion-VAS checkpoints finetuned on SAIL-VOS on Hugging Face. To download them, run:

```bash
mkdir checkpoints
cd checkpoints
git lfs install
git clone https://huggingface.co/kaihuac/diffusion-vas-amodal-segmentation
git clone https://huggingface.co/kaihuac/diffusion-vas-content-completion
cd ..
```
*Note: Ignore any Windows-related warnings when downloading.*

For **Depth Anything V2**'s checkpoints, download the Pre-trained Models (e.g., Depth-Anything-V2-Large) from [this link](https://github.com/DepthAnything/Depth-Anything-V2) and place them inside the `checkpoints/` folder.

### Inference

To run inference, simply execute:

```bash
python demo.py
```

This will infer the birdcage example from `demo_data/`.

To try different examples, modify the `seq_name` argument:

```bash
python demo.py --seq_name <your_sequence_name>
```

You can also change the checkpoint path, data output paths, and other parameters as needed.

### Using custom data

Start with a video, use the **SAM2**'s [web demo](https://sam2.metademolab.com/) or its [codebase](https://github.com/facebookresearch/sam2) to segment the target object, and extract frames preferably at 8 FPS. Ensure that the output follows the same directory structure as examples from `demo_data/` before running inference.

## Evaluation

We currently support evaluation on **SAIL-VOS-2D** and **TAO-Amodal**.

### 1. Download Datasets

Download [SAIL-VOS-2D](https://sailvos.web.illinois.edu/_site/index.html) and [TAO-Amodal](https://huggingface.co/datasets/chengyenhsieh/TAO-Amodal) by following their official instructions.

Additionally, download [our](https://huggingface.co/datasets/kaihuac/diffusion_vas_datasets/tree/main) curated annotations and precomputed evaluation results:

```bash
git clone https://huggingface.co/datasets/kaihuac/diffusion_vas_datasets
```

This includes:
- `diffusion_vas_sailvos_train.json`
- `diffusion_vas_sailvos_val.json`
- `diffusion_vas_tao_amodal_val.json`
- `tao_amodal_track_ids_abs2rel_val.json`
- `sailvos_complete_objs_as_occluders.json`
- Precomputed `eval_outputs/` folder

### 2. Generate Evaluation Results

To evaluate the model, first generate result files using the scripts below. **Alternatively**, you can skip this step and directly use our precomputed results in `eval_outputs/`.

*Note: Please replace the paths in the commands with your own dataset and json annotation paths.*

**SAIL-VOS-2D**
```bash
cd eval
python eval_diffusion_vas_sailvos.py \
    --eval_data_path /path/to/SAILVOS_2D/ \
    --eval_annot_path /path/to/diffusion_vas_sailvos_val.json \
    --eval_output_path /path/to/eval_outputs/
```

**TAO-Amodal**
```bash
python eval_diffusion_vas_tao_amodal.py \
    --eval_data_path /path/to/TAO/frames/ \
    --eval_annot_path /path/to/diffusion_vas_tao_amodal_val.json \
    --track_ids_path /path/to/tao_amodal_track_ids_abs2rel_val.json \
    --eval_output_path /path/to/eval_outputs/
```

### 3. Compute Metrics

Once the result files are ready, run the metric scripts:

**SAIL-VOS-2D**
```bash
python metric_diffusion_vas_sailvos.py \
    --eval_data_path /path/to/SAILVOS_2D/ \
    --eval_annot_path /path/to/diffusion_vas_sailvos_val.json \
    --pred_annot_path /path/to/eval_outputs/diffusion_vas_sailvos_eval_results.json
```

**TAO-Amodal**
```bash
python metric_diffusion_vas_tao_amodal.py \
    --eval_data_path /path/to/TAO/frames/ \
    --eval_annot_path /path/to/diffusion_vas_tao_amodal_val.json \
    --track_ids_path /path/to/tao_amodal_track_ids_abs2rel_val.json \
    --pred_annot_path /path/to/eval_outputs/diffusion_vas_tao_amodal_eval_results.json
```

## Finetuning on SAIL-VOS
We currently support fine-tuning for both the amodal segmentation and content completion stages on SAIL-VOS, based on [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) and adapted from [SVD Xtend](https://github.com/pixeli99/SVD_Xtend).

*Note: Please replace the paths in the commands with your own dataset and annotation paths. The json annotations can be downloaded as shown in the Evaluation section.*

**Amodal segmentation fine-tuning**

We provide end-to-end fine-tuning conditioned on modal masks and depth maps. The training script is:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch train/train_diffusion_vas_amodal_segm.py \
    --data_path /path/to/SAILVOS_2D/ \
    --train_annot_path /path/to/diffusion_vas_sailvos_train.json \
    --eval_annot_path /path/to/diffusion_vas_sailvos_val.json \
    --output_dir /path/to/train_diffusion_vas_amodal_seg_outputs
```

*Note*:
* Our default implementation runs the depth estimator during each training step, which requires more than 24GB memory per GPU and significantly increases training time (~120 hours on 8× A6000s).
* To reduce memory usage and training time, we highly recommend precomputing and saving pseudo-depth maps before training. This allows training on RTX 3090s and reduces training time (~30 hours) considerably.

**Content completion fine-tuning**

We provide end-to-end fine-tuning conditioned on modal RGB images and predicted amodal masks:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch train/train_diffusion_vas_content_comp.py \
    --data_path /path/to/SAILVOS_2D/ \
    --train_annot_path /path/to/sailvos_complete_objs_as_occluders.json \
    --eval_annot_path /path/to/sailvos_complete_objs_as_occluders.json \
    --occluder_data_path /path/to/sailvos_complete_objs_as_occluders.json \
    --output_dir /path/to/train_diffusion_vas_content_comp_outputs
```

This stage does not require depth estimation, and training typically completes in ~30 hours on 8× RTX 3090s.

## Acknowledgement

This work builds on top of several excellent projects, including [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), [SAM2]((https://github.com/facebookresearch/sam2)), [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt), and [SVD Xtend](https://github.com/pixeli99/SVD_Xtend). Our training and evaluation are based on [SAIL-VOS](https://sailvos.web.illinois.edu/_site/index.html) and [TAO-Amodal](https://huggingface.co/datasets/chengyenhsieh/TAO-Amodal). We sincerely thank the authors for their contributions.


## Citation

If you find this work helpful, please consider citing our papers:

```bibtex
@InProceedings{chen2025diffvas,
    author    = {Chen, Kaihua and Ramanan, Deva and Khurana, Tarasha},
    title     = {Using Diffusion Priors for Video Amodal Segmentation},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {22890-22900}
}
```
```bibtex
@article{hsieh2023taoamodal,
  title={TAO-Amodal: A Benchmark for Tracking Any Object Amodally},
  author={Cheng-Yen Hsieh and Kaihua Chen and Achal Dave and Tarasha Khurana and Deva Ramanan},
  journal={arXiv preprint arXiv:2312.12433},
  year={2023}
}
```


