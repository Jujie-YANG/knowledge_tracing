# knowledge_tracing

[Knowledge Tracing(paperswithcode)](https://paperswithcode.com/task/knowledge-tracing): Knowledge Tracing is the task of modelling student knowledge over time so that we can accurately predict how students will perform on future interactions. Improvement on this task means that resources can be suggested to students based on their individual needs, and content which is predicted to be too easy or too hard can be skipped or delayed.

## Dataset
- [ASSISTmentsData](https://sites.google.com/site/assistmentsdata/datasets)
- [Kaggle: Riiid Answer Correctness Prediction](https://www.kaggle.com/competitions/riiid-test-answer-prediction/data)

## Kaggle Materials: 
- Competition:
[Riiid AIEd Challenge 2020](https://www.kaggle.com/competitions/riiid-test-answer-prediction/overview): In this competition, your challenge is to create algorithms for "Knowledge Tracing," the modeling of student knowledge over time. The goal is to accurately predict how students will perform on future interactions. You will pair your machine learning skills using Riiid’s EdNet data.

- notebook:
    - [A self-Attentive model for Knowledge Tracing](https://www.kaggle.com/code/wangsg/a-self-attentive-model-for-knowledge-tracing/notebook)

## Conference:
- [CHI 2023](https://chi2023.acm.org/)

## Requirement:
- ### Packages:
    - torch
    - pytorch-lightning
    - pandas
    - numpy
    - scikit-learn

## Problems:
- ```brew install ipython```
```
Traceback (most recent call last):
  File "/Users/yangjujie/github_repo/knowledge_tracing/train.py", line 3, in <module>
    from dataset import get_dataloaders
  File "/Users/yangjujie/github_repo/knowledge_tracing/dataset.py", line 1, in <module>
    from torch.utils.data import Dataset, DataLoader
  File "/Users/yangjujie/opt/anaconda3/lib/python3.8/site-packages/torch/__init__.py", line 202, in <module>
    from torch._C import *  # noqa: F403
ImportError: dlopen(/Users/yangjujie/opt/anaconda3/lib/python3.8/site-packages/torch/_C.cpython-38-darwin.so, 0x0002): Library not loaded: '@loader_path/libshm.dylib'
  Referenced from: '/Users/yangjujie/opt/anaconda3/lib/python3.8/site-packages/torch/lib/libtorch_python.dylib'
  Reason: tried: '/Users/yangjujie/opt/anaconda3/lib/python3.8/site-packages/torch/lib/libshm.dylib' (no such file), '/usr/local/lib/libshm.dylib' (no such file), '/usr/lib/libshm.dylib' (no such file)
  ```

```
To restart postgresql@14 after an upgrade:
  brew services restart postgresql@14
Or, if you don't want/need a background service you can just run:
  /opt/homebrew/opt/postgresql@14/bin/postgres -D /opt/homebrew/var/postgres
==> python@3.9
Python has been installed as
  /opt/homebrew/bin/python3.9

Unversioned and major-versioned symlinks `python`, `python3`, `python-config`, `python3-config`, `pip`, `pip3`, etc. pointing to
`python3.9`, `python3.9-config`, `pip3.9` etc., respectively, have been installed into
  /opt/homebrew/opt/python@3.9/libexec/bin

You can install Python packages with
  pip3.9 install <package>
They will install into the site-package directory
  /opt/homebrew/lib/python3.9/site-packages

tkinter is no longer included with this formula, but it is available separately:
  brew install python-tk@3.9

If you do not need a specific version of Python, and always want Homebrew's `python3` in your PATH:
  brew install python3

See: https://docs.brew.sh/Homebrew-and-Python
==> jupyterlab
Additional kernels can be installed into the shared jupyter directory
  /opt/homebrew/etc/jupyter

zsh completions have been installed to:
  /opt/homebrew/share/zsh/site-functions
```

## Interpreter
python 3.9.14 64-bit /opt/homebrew/bin/python3.9 Global
install Python packages with ```pip3.9 install <package>```