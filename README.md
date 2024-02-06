# Visual_Question_Answering
## 1. Introduction
**Visual Question Answering (VQA)** is a common problem in Machine Learning, applying techniques related to two fields of Computer Vision and Natural Language Processing. The core concept of this problem is to analyze an image and answer a question about that image. The first step is to analyze the input information, including using image processing and natural language processing techniques to handle the question posed. Then, the VQA system will combine the information obtained from the image analysis and the context of the question to create a suitable answer. Therefore, a program with high accuracy needs to build well both of these components, posing a great challenge in solving the problem of question answering with images.

<p align="center">
  <img width="800" alt="VQA System" src="https://github.com/dinhquy-nguyen-1704/Visual_Question_Answering/assets/127675330/f89c09bb-8a9c-4088-86b8-0df1461e39a2">
</p>
<p align="center">

In this project, I will build a VQA program using Image Encoders (CNN, ViT, CLIP) for images and Text Encoders (LSTM, RoBERTa, CLIP) for natural language processing. The input and output of the program are as follows: 
* Input: A pair of image and question in natural language.
* Output: An answer to the question about the image (Yes/No).

## 2. Dataset
You can download the vqa-coco-dataset [here](https://drive.google.com/file/d/1kc6XNqHZJg27KeBuoAoYj70_1rT92191/view). After that, you should organize the folder structure as follows:

- 📁 Visual_Question_Answering
  - 📁 data
    - 📂 val2014-resised
    - 📄 vaq2.0.TrainImages.txt
    - 📄 vaq2.0.DevImages.txt
    - 📄 vaq2.0.TestImages.txt
  - 📁 CLIP
  - 📁 CNN_LSTM
  - 📁 ViT_RoBERTa
  - 📁 src
  - 🐍 main_CLIP.py
  - 🐍 main_CNN_LSTM.py
  - 🐍 main_ViT_RoBERTa.py
  - 📄 README.md

<p align="center">
  <img width="800" alt="VQA Dataset" src="https://github.com/dinhquy-nguyen-1704/Visual_Question_Answering/assets/127675330/c7374e70-ea97-4c03-aaba-6e2157145c5f">
</p>
<p align="center">
  <em>Some sample data in the VQA dataset in the form of Yes/No questions</em>
</p>

## 3. Train models
First, clone this repo and organize the data as above.
```
git clone https://github.com/dinhquy-nguyen-1704/Visual_Question_Answering.git
cd Visual_Question_Answering
```
### 3.1. Requirements
```
pip install timm
pip install transformers
pip install open_clip_torch
```
### 3.2. CNN - LSTM
If you want to use CNN as Image Encoder and LSTM for Text Encoder as well as classifier:
```
python main_CNN_LSTM.py --cnn_model_name resnet50
```
### 3.3. ViT - RoBERTa
If you want to use ViT as Image Encoder, RoBERTa for Text Encoder and LSTM as classifier:
```
python main_ViT_RoBERTa.py --img_feature_extractor_name google/vit-base-patch16-224 --text_tokenizer_name roberta-base
```
### 3.4. CLIP
If you want to use CLIP as the Encoders and MLP for the classifier:
```
python main_ViT_RoBERTa.py --clip_model_type ViT-B-32 --clip_pretrained laion2b_e16
```

## 4. Result
The metric used in this task is accuracy, the result is evaluated on the Test set of the Dataset.
| Image Encoder  | Text Encoder | Accuracy |
|----------------|--------------|----------|
| CNN            | LSTM         | 54%      |
| ViT            | RoBERTa      | 63%      |
| CLIP           | CLIP         | 73%      |
