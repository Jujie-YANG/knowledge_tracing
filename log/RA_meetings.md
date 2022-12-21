# Logs of meetings with Zhaoxing
## 12.20
- Paper points: (What to do tomorrow)
    1. Why LSTM not other RNN models: Memory? Recent Importance.
    2. Longer Sequence: voca_size -> 2 (Last query+RNN particular subject based on their previous interactions)
    3. Distinguish qkv positional embedding
- LSTM for EKT: One of the benefits of using an LSTM RNN for explainable knowledge tracing is that it is able to take into account the temporal dependencies in the data, meaning that it can consider not just the current state of the student's knowledge, but also how their knowledge has evolved over time. This can help to provide a more nuanced and accurate prediction of the student's knowledge.
- Rasch-model based embedding
- Forgetting mechanism
- RNN vs BERT: Recurrent neural networks (RNNs) and BERT (Bidirectional Encoder Representations from Transformers) are both types of neural network architectures that can be used for knowledge tracing, but they differ in their specific capabilities and the types of tasks they are best suited for.

    RNNs, including LSTM (Long Short-Term Memory) RNNs, are particularly well-suited for modeling sequential data and predicting the next item in a sequence based on the items that come before it. This makes them well-suited for tasks such as language modeling, where the goal is to predict the next word in a sentence based on the words that come before it. In the context of knowledge tracing, an RNN could be used to predict a student's knowledge of a particular subject based on their previous interactions with learning materials.

    BERT, on the other hand, is a type of transformer-based language model that is particularly well-suited for natural language processing tasks, such as language understanding and machine translation. BERT uses a technique called self-attention to analyze the relationships between words in a sentence, allowing it to better understand the meaning of the sentence as a whole. BERT can be fine-tuned for specific tasks, such as question answering or text classification, by training it on a large dataset of labeled examples for the task.

    In general, RNNs are better suited for tasks involving sequential data, while BERT is better suited for tasks involving natural language processing. The best approach for a given knowledge tracing task will depend on the specific needs and constraints of the problem at hand.

## 12.18
- Find other techniques that are added to DKT(transformers)
    - [Code: SAINT PyTorch implementation](https://github.com/arshadshk/SAINT-pytorch) - [Paper: Towards an Appropriate Query, Key, and Value Computation for Knowledge Tracing](https://arxiv.org/pdf/2002.07033.pdf) 
    - [Code: Implementation of SAINT+: Integrating Temporal Features for EdNet Correctness Prediction](https://github.com/Shivanandmn/SAINT_plus-Knowledge-Tracing-) - [PaperWithCode:SAINT/SAINT+](https://paperswithcode.com/paper/towards-an-appropriate-query-key-and-value)
    - [Code: DSAKT](https://github.com/Fusion4233919/DSAKT) - [PaperWithCode](https://paperswithcode.com/paper/application-of-deep-self-attention-in): Same achitecture as SAINT
    - [Context-Aware Attentive Knowledge Tracing](https://paperswithcode.com/paper/context-aware-attentive-knowledge-tracing): AKT uses a novel monotonic attention mechanism that relates a learner’s future responses to assessment questions to their past responses; attention weights are computed using exponential decay and a context-aware relative distance measure, in addition to the similarity between questions. Moreover, we use the Rasch model to regularize the concept and question embeddings; these embeddings are able to capture individual differences among questions on the same concept without using an excessive number of parameters
    - [Last Query Transformer RNN for knowledge tracing](https://paperswithcode.com/paper/last-query-transformer-rnn-for-knowledge): Basically, I use both transformer encoder and RNN to deal with time series input. The novel point of the model is that it only uses the last input as query in transformer encoder, instead of all sequence, which makes QK matrix multiplication in transformer Encoder to have O(L) time complexity, instead of O(L^2).

There are several different approaches to knowledge tracing, and the specific type of embedding that you use may depend on the specific needs of your application. Some common options for creating embeddings for knowledge tracing include:
1. Latent variable models: These models represent knowledge as a latent (hidden) variable that can be inferred from student responses to a series of items or questions. Examples of latent variable models include the Rasch model, which I mentioned earlier, as well as other models such as the item response theory (IRT) model.
2. Neural network models: These models use artificial neural networks to learn representations of knowledge from student responses to a series of items or questions. One popular approach is to use a recurrent neural network (RNN) to model the temporal dynamics of student learning.
3. Matrix factorization techniques: These techniques factorize a matrix of student responses into a low-dimensional representation, which can be used to identify patterns in the data and make predictions about future student performance. Examples of matrix factorization techniques include singular value decomposition (SVD) and non-negative matrix factorization (NMF).

## 12.16
- [HawkesKT code](https://github.com/THUwangcy/HawkesKT)
- [Google AI 2018 BERT pytorch implementation](https://github.com/codertimo/BERT-pytorch)

## 12.13
- For the SAKT model, it should calculate accuracy of predicting last value of output sequence. (Look through Kaggle code notebook to see how they calculate the acc, loss, and auc)
- Try to understand [MonaCoBert](https://github.com/codingchild2424/MonaCoBERT)
    - The inputs of the model?
    - The dataloader, dataset it used?
- How does someone do with SAKT algorithms (Below are three repo using SAKT)
    - [shalini1194/SAKT](https://github.com/shalini1194/SAKT/tree/master/2019-EDM)
    - [Simple and performant implementations of learner performance prediction algorithms](https://github.com/theophilegervet/learner-performance-prediction)
    - [jdxyw/deepKT](https://github.com/jdxyw/deepKT)
- Look through Rasch model([Context-Aware Attentive Knowledge Tracing](https://dl.acm.org/doi/pdf/10.1145/3394486.3403282)) and Hawkes process ([Temporal Cross-Effects in Knowledge Tracing](https://dl.acm.org/doi/pdf/10.1145/3437963.3441802?casa_token=O7ucSempjVMAAAAA:wINOUz7ts87gTvSM57Xi5motZ9hApjn-vQAQQaupDF5xer8HBXQwMw78WE2JOXR_Ts9m0oAeaJM)) to add the forgetting mechanism 
- Ablation Study, Heatmap (different algorithms, probability of ) 
    - Example1:[MonaCoBert](https://github.com/codingchild2424/MonaCoBERT))
    - Example2:[SAKT](https://arxiv.org/pdf/1907.06837.pdf)
    - Example3:[Deep Graph Memory Networks for Forgetting-Robust Knowledge Tracing](https://arxiv.org/pdf/2108.08105.pdf): Graph will not needed in our experiment

## 12.7
- Models to add before BERT:
    - [Deep Graph Memory Networks for Forgetting-Robust Knowledge Tracing](https://arxiv.org/pdf/2108.08105.pdf): AKT [8]: This model combines an attention model with Rasch model-based embeddings, which exponentially decays attention weights w.r.t. the distance of questions in a sequence in order to account for student’s forgetting effect.
    - [Temporal Cross-Effects in Knowledge Tracing](https://dl.acm.org/doi/pdf/10.1145/3437963.3441802?casa_token=O7ucSempjVMAAAAA:wINOUz7ts87gTvSM57Xi5motZ9hApjn-vQAQQaupDF5xer8HBXQwMw78WE2JOXR_Ts9m0oAeaJM): HawkesKT [23]: This model uses the Hawkes process [26] to count for a student’s forgetting behavior.


## 11.2
- Things to do in the lab:
    - AIED theme: "AI in Education for Sustainable Society", feature selection may close to accessibility of disabled people
    - password: yyy (yeah yeah yeah)
    - Build virtual env: [conda create -n pytorchenv python=3.9 anaconda](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/) - conda activate pyorchenv / conda deactivate - if activate correctly: ```(pytorchenv) C:\User\..```

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
