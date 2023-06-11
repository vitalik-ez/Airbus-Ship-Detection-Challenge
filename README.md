# Airbus-Ship-Detection-Challenge
## UNet

This implementation of [UNet](https://arxiv.org/pdf/1505.04597v1.pdf) differs from the original one in the following ways:
1. It does not take cropping into consideration while concatenating the feature maps from the contracting and expanding paths.
2. It incorporates padding in the contracting path. This ensures that the output feature map is the same size as the input image. This is significant because the majority of modern semantic segmentation architectures adhere to this rule.

The model was trained on only 5 epochs (input size 512x512). This process took approximately 5 hours (Tesla T4 Colab). To improve the accuracy of the model, you need to train on more epochs.  To speed up the inference, you can try to convert the model to TensorRT format (the conversion should be done on the computer that will be running the model)

In addition to the Unet model, the [Swin-Transformer-Semantic-Segmentation](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation) ([paper](https://arxiv.org/pdf/2103.14030.pdf)) and [PIDNet](https://github.com/XuJiacong/PIDNet) ([paper](https://arxiv.org/pdf/2206.02066.pdf)) models are also being trained.  Training models requires powerful graphics cards. For me, the only solution is to use Colab or Kaggle, but there are limitations for the free version. When the models are trained, I will add them here and make a comparison table.   
Alternatively, you can split the input frame into tiles (for example, 250x250), which will improve the detection of small ships [SAHI: Slicing Aided Hyper Inference](https://github.com/obss/sahi)  

## Usage

Download the dataset to the input folder (please follow the instructions described in the README file located in the same folder). This dataset is already prepared and split into train (80%) valid (20%) for model training. The **prepare_dataset.ipynb** notebook located in the **src** folder contains all the necessary code to transform the raw dataset from Kaggle into the dataset used in this project.   
The output folder contains the training results (weights, graphics). Trained weights placed on [google drive](https://drive.google.com/drive/folders/1O-3RmJLtXR-lN9ig5u0tNOL6mYpPyfLA?usp=sharing).   

Installation:
```bash
pip install -r requirements.txt
```
Training:
```bash
cd src/
python train.py --epochs 125 --batch 4 --lr 0.005
```

Inference:
```bash
cd src/
python inference_image.py --model ../outputs/best_model_iou.pth --input ../input/test_images/ --imgsz 512
```
