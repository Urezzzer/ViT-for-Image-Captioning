# ViT-for-Image-Captioning
Implementation of Vision Transformer to solve image captioning task, a simple way to achieve SOTA, in Pytorch

* Image captioning model based on Image2Seq Architecture (CNN Feature Extractor + LSTM with MultiHead Attention)
Basic model to understand task. I created model, you can use/develop it for free :) Easy to implement. Fast to train. Fun to caption.

* ViT for Masked-Image Modeling
Plan to use it as feature extractor in Image2Seq and in Full Transformer Architectures.

Trained on COCO 2017 Dataset - https://cocodataset.org/#home

Size of dataset is about 117k. 

I created unique pairs: Image - Caption (on average 5 Captions per Image). 

So I have 591753 objects in train dataset and 25014 objects in validation dataset but I used first 20k objects to train models and 10k to validate them (I don't have the ability to use more data).

Model | CNN + LSTM with Attn | ViT + LSTM with Attn | ViT + GPT |
--- | --- | --- | --- |
Train BLEU-1 | 0 | 0 | 0 |
--- | --- | --- | --- |
Train BLEU-2 | 0 | 0 | 0 |
--- | --- | --- | --- |
Train BLEU-3 | 0 | 0 | 0 |
--- | --- | --- | --- |
Train BLEU-4 | 0 | 0 | 0 |
--- | --- | --- | --- |
Valid BLEU-1 | 0 | 0 | 0 |
--- | --- | --- | --- |
Valid BLEU-2 | 0 | 0 | 0 |
--- | --- | --- | --- |
Valid BLEU-3 | 0 | 0 | 0 |
--- | --- | --- | --- |
Valid BLEU-4 | 0 | 0 | 0 |
