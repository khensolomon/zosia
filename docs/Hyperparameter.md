# Understanding the Default Hyperparameters

These values in the `nmt.py` script are the starting point for training the Transformer model. They control the model's architecture (its size and complexity) and the training process (how it learns from the data).

* **`DEFAULT_EPOCHS = 150`**
  An **epoch** is one complete pass through the entire training dataset. The script will train for a maximum of 150 epochs, but it may stop earlier if performance on the validation set stops improving (due to the `patience` setting).

* **`DEFAULT_BATCH_SIZE = 32`**
  Instead of processing sentences one by one, the script groups them into **batches** for efficiency. It will process 32 sentences at a time.

* **`DEFAULT_LR = 0.0001`**
  The **learning rate** controls how much the model adjusts its internal parameters after each batch. A smaller value like this generally leads to slower but more stable learning.

* **`DEFAULT_EMBED_SIZE = 256`**
  This is the **embedding size**. Each word in the vocabulary is represented by a vector of 256 numbers. This vector captures the word's "meaning" within the model's vector space.

* **`DEFAULT_NHEAD = 4`**
  The Transformer model uses "multi-head attention," allowing it to analyze the sentence from multiple perspectives simultaneously. This setting configures it to use **4 "heads"** or perspectives.

* **`DEFAULT_FF_HID_DEN_SIZE = 512`**
  This is the size of the internal feed-forward network layer within each Transformer block. It's a key part of how the model processes information after the attention step.

* **`DEFAULT_NUM_LAYERS = 3`**
  This controls the model's depth. The script will stack **3 Transformer blocks** on top of each other in both the encoder (which reads the source sentence) and the decoder (which generates the target sentence).

* **`DEFAULT_DROPOUT = 0.1`**
  A regularization technique to prevent overfitting. During training, it will randomly "drop out" or ignore **10%** of the neurons, forcing the model to learn more robust and generalizable patterns.

* **`DEFAULT_VOCAB_SIZE = 8000`**
  This instructs the tokenizer to build a vocabulary of the **8,000 most frequent** words or sub-word pieces from the training data.

* **`DEFAULT_VAL_SPLIT = 0.15`**
  This reserves **15%** of the training data to be used as a validation set. The model does not learn from this data; it only uses it to evaluate its performance after each epoch to check for improvement.

* **`DEFAULT_PATIENCE = 10`**
  This enables **early stopping**. If the model's performance on the validation set does not improve for 10 consecutive epochs, the training process will halt automatically. This saves time and prevents the model from overfitting.