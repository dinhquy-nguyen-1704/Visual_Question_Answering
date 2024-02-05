# Visual_Question_Answering
## Introduction
**Visual Question Answering (VQA)** is a common problem in Machine Learning, applying techniques related to two fields of Computer Vision and Natural Language Processing. The core concept of this problem is to analyze an image and answer a question about that image. The first step is to analyze the input information, including using image processing and natural language processing techniques to handle the question posed. Then, the VQA system will combine the information obtained from the image analysis and the context of the question to create a suitable answer. Therefore, a program with high accuracy needs to build well both of these components, posing a great challenge in solving the problem of question answering with images.

![Visual Question Answering](https://github.com/dinhquy-nguyen-1704/Visual_Question_Answering/assets/127675330/56ed3b61-b9ae-4ad8-9fa7-ed5e6f285bce)

In this project, I will build a VQA program using Image Encoders (CNN, ViT, CLIP) for images and Text Encoders (LSTM, RoBERTa, CLIP) for natural language processing. The input and output of the program are as follows: 
* Input: A pair of image and question in natural language.
* Output: An answer to the question about the image (yes/no question).

## Dataset
You can download vqa-coco-dataset [here](https://drive.google.com/file/d/1kc6XNqHZJg27KeBuoAoYj70_1rT92191/view). After that, you should organize the folder structure as follows:

- 📁 Visual_Question_Answering
  - 📁 data
    - 📂 val2014-resesed
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

![VQA Dataset](https://github.com/dinhquy-nguyen-1704/Visual_Question_Answering/assets/127675330/c7374e70-ea97-4c03-aaba-6e2157145c5f)
