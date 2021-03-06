# ViT-for-Image-Captioning
Implementation of Vision Transformer to solve image captioning task, a simple way to achieve SOTA, in Pytorch

* [Image captioning model based on Image2Seq Architecture (CNN Feature Extractor + LSTM with MultiHead Attention)](./image2seq.ipynb)

Basic model to understand image captioning task. I created the model, you can use/develop it for free :)

Several examples of captioning:

![](./examples/1.jpg)
---
![](./examples/2.jpg)
---
![](./examples/3.jpg)
---
![](./examples/4.jpg)
---
Model has motorcycle bias so  it describes only vehicles if sees them on the picture.

![](./examples/5.jpg)
---
![](./examples/6.jpg)
---

* ViT for Masked-Image Modeling in Image2Seq

### in progress

* ViT for Masked-Image Modeling in Full Transformer Architecture.

Trained on [COCO 2017 Dataset](https://cocodataset.org/#home)

Size of dataset is about 117k. 

I created unique pairs: Image - Caption (on average 5 Captions per Image). 

So I got 591753 objects in train dataset and 25014 objects in validation dataset but I used random-sampled 61k objects per epoch to train models and first 10k objects from validation dataset to validate them. Overall 35 epochs (I don't have the opportunity to train anymore. the same's about number of train objects).

Model | CNN + LSTM with Attn | ViT + LSTM with Attn | ViT + GPT |
--- | --- | --- | --- |
Train BLEU-1 | 0.568 | 0 | 0 |
Train BLEU-2 | 0.414 | 0 | 0 |
Train BLEU-3 | 0.302 | 0 | 0 |
Train BLEU-4 | 0.230 | 0 | 0 |
Valid BLEU-1 | 0.468 | 0 | 0 |
Valid BLEU-2 | 0.336 | 0 | 0 |
Valid BLEU-3 | 0.248 | 0 | 0 |
Valid BLEU-4 | 0.201 | 0 | 0 |

[Inspired](https://github.com/lucidrains/vit-pytorch) me for this project!
