# Learning Materials
## Learning materials about Attention
- ### Articles:
    - [PYTORCH-TRANSFORMERS](https://pytorch.org/hub/huggingface_pytorch-transformers/):PyTorch implementations of popular NLP Transformers
    
- ### Tutorials:
    - [Bilibili李宏毅 Self-Attention视频讲解(2 episodes)](https://www.bilibili.com/video/BV1J94y1f7u5?p=26)


## Learning materials about Self Supervised Learning(SSL)
- ### Articles:

- ### Tutorials:
    - [Bilibili李宏毅Self supervised learning视频讲解(4 episodes)](https://www.bilibili.com/video/BV1J94y1f7u5?p=46)


## Learning materials about BERT
- ### Blogs:
    - [Hugging Face: BERT](https://huggingface.co/blog/bert-101)
    - [BERT Explained: State of the art language model for NLP(2018)](https://towardsdatascience.com/bert-explained-state-of-the-art-language-model-for-nlp-f8b21a9b6270)
- ### Tutorials:
    - [Bilibili李宏毅BERT视频讲解](https://www.bilibili.com/video/BV1J94y1f7u5/?p=50)
    - [Bert fine-tune tutorials](https://mccormickml.com/2019/07/22/BERT-fine-tuning/) + [Part2 training and validation](https://www.youtube.com/watch?v=Hnvb9b7a_Ps)
    
- ### GitHub repo:
    - [Google: Albert](https://github.com/google-research/albert): A Lite BERT for Self-supervised Learning of **Language Representations**
    - [Google: TensorFlow code and pre-trained models for BERT](https://github.com/google-research/bert)
    - [Google AI 2018 BERT pytorch implementation](https://github.com/codertimo/BERT-pytorch)
    
- ### Other two BERT applications in KT
    - [MathBERT: A Pre-trained Language Model for General NLP Tasks in Mathematics Education](https://paperswithcode.com/paper/mathbert-a-pre-trained-language-model-for) - Tensorflow
    - [MonaCoBERT: Monotonic attention based ConvBERT for Knowledge Tracing](https://paperswithcode.com/paper/monacobert-monotonic-attention-based-convbert) - Pytorch


## Small points
- [【Python】 垃圾回收机制和gc模块](https://www.cnblogs.com/franknihao/p/7326849.html): # gc.collect() 返回处理这些循环引用一共释放掉的对象个数
- [argparse — Parser for command-line options, arguments and sub-commands](https://docs.python.org/3/library/argparse.html)

- Loss: reduction (string, optional) – Specifies the reduction to apply to the output
    - Training: ```loss = lossFunc(task1_pre, task1_gold)```:[nn.BCELoss(reduction='sum')](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)
    - PreTraining: ```loss = multi_loss(pre_mask_q, gold_q, pre_mask_s, gold_s, pre_dif_q, gold_dif_q)```: 
        - mask_q_loss & mask_s_loss = ```F.cross_entropy(pre_q, gold_q, ignore_index=Constants.PAD, reduction='mean')``` ([nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html))
        - dif_loss = mse(pre_dif, gold_dif) ```nn.MSELoss(reduction='mean')```([nn.MSELoss(reduction='mean')](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html))
        - loss = mask_q_loss + mask_s_loss + 30*dif_loss
- [torch.nn.Embedding explained (+ Character-level language model)](https://www.youtube.com/watch?v=euwN5DHfLEo&ab_channel=mildlyoverfitted)


## Others
- ### Blogs:
    - [This repo is built for paper: Attention Mechanisms in Computer Vision: A Survey paper](https://github.com/MenghaoGuo/Awesome-Vision-Attentions)
    - [FightingCV 代码库， 包含 Attention,Backbone, MLP, Re-parameter, Convolution](https://github.com/xmu-xiaoma666/External-Attention-pytorch)

- ### Tutorials
    - [More Bilibili tutorials regarding DL/ML given by 李宏毅 including GPT-3, GAN, SSL...](https://www.bilibili.com/video/BV1J94y1f7u5?p=50&vd_source=4e20016bd1355fe9ad9e32194a97d42a)
        
    
