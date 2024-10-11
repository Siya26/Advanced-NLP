# Advanced NLP
## Assignment 1

The code for each question is written in a separate python file. The deep learning framework used throughout the assignment is **Pytorch**.

Each python file contains the code for training, testing and calculating perplexity scores for the respective model.

The pre-trained embeddings used for this assignment can be found at `glove.6B.100d.txt`.

Link to the saved models: https://iiitaphyd-my.sharepoint.com/:f:/g/personal/siya_puttagunta_students_iiit_ac_in/ElkyY_5fb_ZPjxu4LHqde9cBm43wyk3lmOH_gz2MwK8Glg?e=haKKw1

## Q1
The file `1.py` contains the code for question 1 which involves the use of NNLM to predict the next word, given a 5-gram context in a sentence.

The model can be trained by running the following command:
```
python3 1.py
```

The train, val and test perplexity scores for NNLM are stored in the following files:
- `2021101062-LM1-train-perplexity.txt`
- `2021101062-LM1-val-perplexity.txt`
- `2021101062-LM1-test-perplexity.txt`

The model is saved at `NNLM.pt`.

## Q2
The file `2.py` contains the code for question 2 which involves the use of LSTM (RNN-based language model) to predict the next word in a sentence.

The model can be trained by running the following command:
```
python3 2.py
```

The train, val and test perplexity scores for LSTM are stored in the following files:
- `2021101062-LM2-train-perplexity.txt`
- `2021101062-LM2-val-perplexity.txt`
- `2021101062-LM2-test-perplexity.txt`

The model is saved at `LSTM.pt`.

## Q3
The file `3.py` contains the code for question 3 which involves the use of Decoder to predict the next word in a sentence.

The model can be trained by running the following command:
```
python3 3.py
```

The train, val and test perplexity scores for Decoder are stored in the following files:
- `2021101062-LM3-train-perplexity.txt`
- `2021101062-LM3-val-perplexity.txt`
- `2021101062-LM3-test-perplexity.txt`

The model is saved at `Decoder.pt`.

