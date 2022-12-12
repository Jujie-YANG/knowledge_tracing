# knowledge_tracing

[Knowledge Tracing(paperswithcode)](https://paperswithcode.com/task/knowledge-tracing): Knowledge Tracing is the task of modelling student knowledge over time so that we can accurately predict how students will perform on future interactions. Improvement on this task means that resources can be suggested to students based on their individual needs, and content which is predicted to be too easy or too hard can be skipped or delayed.

BERT is a powerful language processing model that can be used for a wide range of natural language processing tasks, but it is not specifically designed for knowledge tracing. Instead, you may want to use a different type of machine learning model, such as a recurrent neural network (RNN) or a long short-term memory (LSTM) model, which are well-suited to analyzing sequential data like student performance data. These models can learn to make predictions about a student's knowledge based on their previous performance, and can be trained using a variety of techniques.

Monotonic attention is a type of attention mechanism that allows a model to focus on different parts of its input in a monotonic fashion, meaning that the model's focus on the input always moves forward and never goes backwards. This can be useful for tasks like machine translation, where the model needs to process the input sentence one word at a time in a specific order.

Span dynamic convolutional attention is a type of attention mechanism that allows a model to focus on different parts of its input using a convolutional neural network. This allows the model to dynamically adjust its focus on the input based on the context of the words, rather than just processing the input in a fixed order. This can be useful for tasks like language modeling, where the model needs to understand the meaning of words in context in order to make accurate predictions.

It is a custom model called SAKT model (Sequential Adaptive Knowledge Tracing) that is designed for knowledge tracing tasks. The SAKT model incorporates the self-attention mechanism used in BERT to better capture the sequential dependencies between student responses to questions.

## Dataset
- [ASSISTmentsData](https://sites.google.com/site/assistmentsdata/datasets)
- [Kaggle: Riiid Answer Correctness Prediction](https://www.kaggle.com/competitions/riiid-test-answer-prediction/data)
  - Data Info:
    - n_skills: 13523
    - index:user_id, len(train_group)=320000, len(valid_group)=73656, train:valid = 8:2
    - input: timestamp(ascending), user_id, content_id(skills), content_type_id(0-questions), answered_correctly

## Model
- Inputs:
  - 'x': student response
  - 'target_id': Content_Id
  - 'label': answer_correctly
  
## CodewithPaper
- [MonaCoBERT: Monotonic attention based ConvBERT for Knowledge Tracing](https://github.com/codingchild2424/MonaCoBERT)
  - ``` 
        # question embedding
        self.emb_q = nn.Embedding(self.num_q, self.hidden_size).to(self.device)
        # response embedding
        self.emb_r = nn.Embedding(self.num_r, self.hidden_size).to(self.device)
        # positional embedding
        self.emb_pid = nn.Embedding(self.num_pid, self.hidden_size).to(self.device)

        self.emb_diff = nn.Embedding(self.num_diff, self.hidden_size).to(self.device)  diff-hardcode = 101?

        self.emb_p = nn.Embedding(self.max_seq_len, self.hidden_size).to(self.device) - p?```

## Kaggle Materials: 
- Competition:
[Riiid AIEd Challenge 2020](https://www.kaggle.com/competitions/riiid-test-answer-prediction/overview): In this competition, your challenge is to create algorithms for "Knowledge Tracing," the modeling of student knowledge over time. The goal is to accurately predict how students will perform on future interactions. You will pair your machine learning skills using Riiid’s EdNet data.

- notebook:
    - [A self-Attentive model for Knowledge Tracing](https://www.kaggle.com/code/wangsg/a-self-attentive-model-for-knowledge-tracing/notebook)
    - [Riiid! SAKT Model - Training - Public](https://www.kaggle.com/code/manikanthr5/riiid-sakt-model-training-public)

## Conference:
- [CHI 2023](https://chi2023.acm.org/)
