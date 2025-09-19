# Image Super-Resolution via Iterative Refinement

[Paper](https://arxiv.org/pdf/2104.07636.pdf ) |  [Project](https://iterative-refinement.github.io/ )

## Brief

This is an unofficial implementation of **Image Super-Resolution via Iterative Refinement(SR3)** by **PyTorch**.

There are some implementation details that may vary from the paper's description, which may be different from the actual `SR3` structure due to details missing. Specifically, we:

- Used the ResNet block and channel concatenation style like vanilla `DDPM`.
- Used the attention mechanism in low-resolution features ( $16 \times 16$ ) like vanilla `DDPM`.
- Encode the $\gamma$ as `FilM` structure did in `WaveGrad`, and embed it without affine transformation.
- Define the posterior variance as $\dfrac{1-\gamma_{t-1}}{1-\gamma_{t}} \beta_t$  rather than $\beta_t$,  which gives similar results to the vanilla paper.

**If you just want to upscale $(64 \times 64)\text{px} \rightarrow (512 \times 512)\text{px}$ images using the pre-trained model, check out [this google colab script](https://colab.research.google.com/drive/1G1txPI1GKueKH0cSi_DgQFKwfyJOXlhY?usp=sharing).**

## Usage
### Environment
```python
pip install -r requirement.txt
```

### Pretrained Model
```python
# Identify the pretrained model and edit [sr|sample]_[ddpm|sr3]_[resolution option].json about "resume_state":
"resume_state": [your pretrained model's path]
```
### Pre-train CNN and generate predicted images

Modify the parameters in several files in the /pretrain_CNN directory, and then run the following script directly.

```shell
python pretrain_CNN/train.py
```

The CNN predictions will be written to the specified path, 
note that the path needs to be specified as the previously generated **dataset/xxx/CNN_sr_[lr]_[hr]**.
### Data Prepare

#### New Start

If you didn't have the data, you can prepare it by following steps:

- [FFHQ 128×128](https://github.com/NVlabs/ffhq-dataset) | [FFHQ 512×512](https://www.kaggle.com/arnaud58/flickrfaceshq-dataset-ffhq)
- [CelebaHQ 256×256](https://www.kaggle.com/badasstechie/celebahq-resized-256x256) | [CelebaMask-HQ 1024×1024](https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view)

Download the dataset and prepare it in **LMDB** or **PNG** format using script.

```python
# Resize to get 16×16 LR_IMGS and 128×128 HR_IMGS, then prepare 128×128 Fake SR_IMGS by bicubic interpolation
python data/prepare_data.py  --path [dataset root]  --out [output root] --size 16,128 -l
```

then you need to change the datasets config to your data path and image resolution: 

```json
# Use config/celebahq.json for updated features with total variations.
"datasets": {
    "train": {
        "dataroot": "dataset/ffhq_16_128", // [output root] in prepare.py script
        "l_resolution": 16, // low resolution need to super_resolution
        "r_resolution": 128, // high resolution
        "datatype": "lmdb", //lmdb or img, path of img files
    },
    "val": {
        "dataroot": "dataset/celebahq_16_128", // [output root] in prepare.py script
        "n_run":1 // can be changed to e.g. 200 to evaluate the model with multiple runs' average
    }
},
"model":{
     "loss": {
            "type": "l1", // l1 or l2
            "TV1_weight": 0.0,//Anisotropic Total Variation Loss Weight, implemented in regularization.py
            "TV2_weight": 0.0,
            "TVF_weight": 0.0, // fractional TV
            "TVF_alpha": 1.6,// fractional TV hyper parameter
            "wavelet_l1_weight": 1.0  //wavelet
        },
        "unet": {
            //....// other configurations of Unet 
            "final_activation": "s_stdleakyrelu", // default swish. Can be chosen from stdrelu, stdleakyrelu, relu, s_stdleakyrelu,and leakyrelu
            "nb_iterations" :10, // number of iterations for the STDReLu/STDReLuLeaky
            "nb_kerhalfsize": 1, // half size of the kernel for the STDReLu/STDReLuLeaky
            "leaky_alpha": 0.2, // alpha for the leaky relu
            "sleaky_beta" : 10.0 // beta for the s_stdleakyrelu
        }

}
```

#### Own Data
You also can use your image data by following steps, and we have some examples in dataset folder.
At first, you should organize the images layout like this, this step can be finished by `data/prepare_data.py` automatically:

```shell
# set the high/low resolution images, bicubic interpolation images path 
dataset/celebahq_16_128/
├── hr_128 # it's same with sr_16_128 directory if you don't have ground-truth images.
├── lr_16 # vinilla low resolution images
└── sr_16_128 # images ready to super resolution
```

```python
# super resolution from 16 to 128
python data/prepare_data.py  --path [dataset root]  --out celebahq --size 16,128 -l
```

*Note: Above script can be used whether you have the vanilla high-resolution images or not.*

then you need to change the dataset config to your data path and image resolution: 

```json
"datasets": {
    "train|val": { // train and validation part
        "dataroot": "dataset/celebahq_16_128",
        "l_resolution": 16, // low resolution need to super_resolution
        "r_resolution": 128, // high resolution
        "datatype": "img", //lmdb or img, path of img files
    }
},
```

### Training/Resume Training

```python
# Use sr.py and sample.py to train the super resolution task and unconditional generation task, respectively.
# Edit json files to adjust network structure and hyperparameters
python sr.py -p train -c config/sr_sr3.json
```

### Test/Evaluation

```python
# Edit json to add pretrain model path and run the evaluation 
python sr.py -p val -c config/sr_sr3.json

# Quantitative evaluation alone using SSIM/PSNR metrics on given result root
python eval.py -p [result root]


#All plots and image grids are produced by the provided shell file. Please open and read that shell file to see exactly what it does and to adjust any paths.
    #What it does (at a glance):
        # a. Concatenates multiple train/val logs into ./plots/*.log
        # b. Calls the plotting scripts to generate comparison figures and CSVs
        # c. Builds evaluation image grids into ./plots/
    #How to use:
        #1.Ensure the plots/ folder exists.
        #2.Read the shell file, update paths/labels and plotting configurations if needed.
        #3.
chmod +x evaluation_plot.sh
#4.
./evaluation_plot.sh
#All outputs will appear under ./plots/.
```


### Inference Alone

Set the  image path like steps in `Own Data`, then run the script:

```python
# run the script
python infer.py -c [config file]
```



## Acknowledgements

Our work is based on the following theoretical works:

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
- [Image Super-Resolution via Iterative Refinement](https://arxiv.org/pdf/2104.07636.pdf)
- [WaveGrad: Estimating Gradients for Waveform Generation](https://arxiv.org/abs/2009.00713)
- [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/abs/1809.11096)

Furthermore, we are benefitting a lot from the following projects:

- https://github.com/bhushan23/BIG-GAN
- https://github.com/lmnt-com/wavegrad
- https://github.com/rosinality/denoising-diffusion-pytorch
- https://github.com/lucidrains/denoising-diffusion-pytorch
- https://github.com/hejingwenhejingwen/AdaFM
