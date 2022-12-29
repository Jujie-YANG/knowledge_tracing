- Notes:
  - Different sequence length:
    - Short sequence(<=100): Rasch + BERT + RNN
    - Long sequence(>100): Rasch + BERT + RNN + last query + last answer 

  - Different Rasch embedding:
    - IRT Rasch embedding is a fixed embedding, which means that the knowledge state of the student is fixed and does not change when estimated.
    - AKT Rasch embedding is a dynamic embedding, which means that the knowledge state of the student is updated when estimated. 

  - Different RNN:
    - LSTM (Long Short-Term Memory) - it considers forgetting mechanism, which is more suitable for the knowledge state of the student
    - GRU (Gated Recurrent Unit) - it doesn't need memory units, so it is more suitable for short sequences. It considers more about the relationship between the current state and the previous state. 

  - The last query and last answer are used to capture the context of the current question.

- Hyperparameters of the model:
  - Batch size: 64
  - Early stopping: 3 epochs. If the validation loss does not decrease for 3 epochs, the training will be stopped. It is used to prevent overfitting and save resources.
  - train/test split: 0.8/0.2
  - embedding size: 128
  - optimizer: Adam, learning rate: 0.001
  - Loss function: BCEWithLogitsLoss (Binary Cross Entropy with Logits Loss)
  - scheduler: OneCycleLR, max learning rate: 0.002
  - Dropout: 0.2
  - Epoch: 10

  - Others:
    - Max sequence length: 100
    - Attention head: 8
    - Hidden size: BERT: 128, FFN: 4 * 128 = 512, RNN: 128
    - Transformer Block: 12
    <!-- - Activation function: GELU (Gaussian Error Linear Unit) -->





  
  
  
  
