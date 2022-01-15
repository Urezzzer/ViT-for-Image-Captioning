# ViT-for-Image-Captioning
Implementation of Vision Transformer to solve image captioning task, a simple way to achieve SOTA, in Pytorch

* Image captioning model based on Image2Seq Architecture (CNN Feature Extractor + LSTM with MultiHead Attention)
Basic model to understand task. I created model, you can use/develop it for free :) Easy to implement. Fast to train. Fun to caption.

* ViT for Masked-Image Modeling
Plan to use it as feature extractor in Image2Seq and in Full Transformer Architectures.

Trained on COCO 2017 Dataset - https://cocodataset.org/#home

Size of dataset is about 117k. 

I created unique pairs: Image - Caption (on average 5 Captions per Image). 

So I have 591753 objects in train dataset and 25014 objects in validation dataset

Model | #1 | #2 | #3 | #4 | #5 | #6 | #7 | #8 | #9 | #10 | #11
--- | --- | --- | --- |--- |--- |--- |--- |--- |--- |--- |---
Seconds | 301 | 283 | 290 | 286 | 289 | 285 | 287 | 287 | 272 | 276 | 269
