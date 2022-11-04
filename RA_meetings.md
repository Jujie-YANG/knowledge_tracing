# Logs of meetings with Zhaoxing
## 11.2
- Things to do in the lab:
    - AIED theme: "AI in Education for Sustainable Society", feature selection may close to accessibility of disabled people
    - password: yyy (yeah yeah yeah)
    - Build virtual env

- Give up [bert_like_model](https://www.kaggle.com/code/luffy521/bert-like-model), because it is not reproducible
- Try to run - the Kaggle notebook: [Riiid! SAKT Model - Training - Public](https://www.kaggle.com/code/manikanthr5/riiid-sakt-model-training-public), and replaced the model with pretrained BERT model

## 10.26
- Run [bert_like_model](https://www.kaggle.com/code/luffy521/bert-like-model)
- Find some other techniques (models) combined with some forgetting machenism or normalizations as a block in front of BERT model

## 10.19
- Two Useful Kaggle Notebooks:
    - [Riiid! SAKT Model - Training - Public](https://www.kaggle.com/code/manikanthr5/riiid-sakt-model-training-public)
    - [bert_like_model](https://www.kaggle.com/code/luffy521/bert-like-model)


## 10.12
- Consider what features Kaggle normally used (EdNet)

## 9.29
- Research Question: Aim to solve cold start problem (small amount of data input)
- Look at [MonaCoBERT: Monotonic attention based ConvBERT for Knowledge Tracing](https://paperswithcode.com/paper/monacobert-monotonic-attention-based-convbert)

## 9.22
- Be more familiar with BERT
    - There is already [a BERT based knowledge tracing experiment:MonaCoBERT](https://paperswithcode.com/paper/monacobert-monotonic-attention-based-convbert)
    - Triangular mask (saint+) -> Monotonic attention (MonaCoBERT)
- Find out how the inputs of the BERT looks like: refer to another RA code

## 9.15
- Main task: 
    - Get familiar with the architecture BERT
        - [Bilibili李宏毅BERT视频讲解](https://www.bilibili.com/video/BV1eV411d7Kp/?vd_source=4e20016bd1355fe9ad9e32194a97d42a)

- Others:
    - Get familiar with [SAINT+](https://github.com/Shivanandmn/SAINT_plus-Knowledge-Tracing-) architecture
    - Find some other pre-train models (fine tuned BERT architecture)

## 9.8
- Determine model structure
    - [SSL(Self-Supervised Learning: pre-training)](https://github.com/Vinci-hp/pretrainKT) + [SAINT+](https://github.com/Shivanandmn/SAINT_plus-Knowledge-Tracing-)
    - And then compare to other SOTA models, e.g. SAINT, AKT, KTM(machine) etc.

## 9.1
- Determine research direction： 
    - [SPAKT: A Self-Supervised Pre-TrAining Method for Knowledge Tracing](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9813711&tag=1) and its corresponding [code here](https://github.com/Vinci-hp/pretrainKT) 
    - [SAINT+: Integrating Temporal Features for EdNet Correctness Prediction](https://paperswithcode.com/paper/saint-integrating-temporal-features-for-ednet) and its corresponding [code here](https://github.com/Shivanandmn/SAINT_plus-Knowledge-Tracing-)

## 8.18 
### Things to do
- Read literatures 
    - Forgetting mechanism:
        - [Deep Graph Memory Networks for Forgetting-Robust Knowledge Tracing](http://arxiv.org/abs/2108.08105)
        - [Augmenting Knowledge Tracing by Considering Forgetting Behavior](https://dl.acm.org/doi/10.1145/3308558.3313565)
    - [SPAKT: A Self-Supervised Pre-TrAining Method for Knowledge Tracing](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9813711&tag=1)
    - [Context-Aware Attentive Knowledge Tracing](https://dl.acm.org/doi/pdf/10.1145/3394486.3403282)
    
    - [SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training](https://paperswithcode.com/paper/saint-improved-neural-networks-for-tabular)
    - [SAINT+: Integrating Temporal Features for EdNet Correctness Prediction](https://paperswithcode.com/paper/saint-integrating-temporal-features-for-ednet)

- Rerun paper code
    - https://github.com/arghosh/AKT (Context-Aware Attentive Knowledge Tracing)
    - https://github.com/Vinci-hp/pretrainKT (SPAKT: A Self-Supervised Pre-TrAining Method for Knowledge Tracing)
    - And code above which is about SAINT and SAINT+

## 8.4 
### Things to do
- Read literatures 
    - Qi Liu, Runze Wu, Enhong Chen, Guandong Xu, Yu Su, Zhigang Chen, and Guoping Hu. Fuzzy cognitive diagnosis for modelling examinee performance. Acm Transactions on Intelligent Systems and Technology, 9(4):1–26, 2018.

    - Ekaterina Vasilyeva, Seppo Puuronen, Mykola Pechenizkiy, and Pekka Rasanen. Feedback adaptation in web-based learning systems. International Journal of Continuing Engineering Education and Life Long Learning, 17(4/5):337, 2007. 
 
    - Meng Wang, Zong Kai Yang, San Ya Liu, Hercy N H Cheng, and Zhi Liu. Using feedback to improve learning: Differentiating between correct and erroneous examples. In International Symposium on Educational Technology, 2015.

    - [A Survey of Knowledge Tracing](https://arxiv.org/pdf/2105.15106.pdf): 4.4 Considering Forgetting after Learning

- Know about some techniques
    - [Decision Transformer](https://proceedings.neurips.cc/paper/2021/file/7f489f642a0ddb10272b5c31057f0663-Paper.pdf)
    - [GAIL](https://proceedings.neurips.cc/paper/2016/file/cc7e2b878868cbae992d1fb743995d8f-Paper.pdf)
