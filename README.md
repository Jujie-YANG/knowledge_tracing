# knowledge_tracing

[Knowledge Tracing(paperswithcode)](https://paperswithcode.com/task/knowledge-tracing): Knowledge Tracing is the task of modelling student knowledge over time so that we can accurately predict how students will perform on future interactions. Improvement on this task means that resources can be suggested to students based on their individual needs, and content which is predicted to be too easy or too hard can be skipped or delayed.

## Dataset
- [ASSISTmentsData](https://sites.google.com/site/assistmentsdata/datasets)
- [Kaggle: Riiid Answer Correctness Prediction](https://www.kaggle.com/competitions/riiid-test-answer-prediction/data)
  - Data Info:
    - n_skills: 13523
    - index:user_id, len(train_group)=320000, len(valid_group)=73656, train:valid = 8:2
    - input: timestamp(ascending), user_id, content_id(skills), content_type_id(0-questions), answered_correctly

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
