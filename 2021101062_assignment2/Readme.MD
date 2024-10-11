# Advanced NLP
## Assignment 2

The deep learning framework used throughout the assignment is **Pytorch**. The file structure is as follows:
- [utils.py](utils.py): Contains the code for preprocessing and functions such as train, test and inference.
- [encoder.py](encoder.py): Contains the encoder class for the transformer which includes positional encoding, multi-head attention, feed-forward layer and encoder layer.
- [decoder.py](decoder.py): Contains the decoder class for the transformer model which consists the decoder layer and the code for generating mask.
- [transformer.py](transformer.py): Contains the transformer class which combines the encoder and decoder classes.
- [train.py](train.py): Contains the code for training the model, calculating train loss and test loss.
- [test.py](test.py): Contains the code for calculating BLEU scores for test sentences.
- [tuning.py](tuning.py): Contains the code for tuning hyperparameters and calculating the best BLEU score.

Link to the trained model `transformer.pt` and the best model `best_model.pt`: https://iiitaphyd-my.sharepoint.com/:f:/g/personal/siya_puttagunta_students_iiit_ac_in/Epy6Qb-sJM5Bohw4Qa8YQIIB254Bi9SWt5VzGZEnVLm2Yg?e=eFgS5w

### Training
To train the model, run the following command:
```
python3 train.py
```
The model is saved at [transformer.pt](transformer.pt).

### Testing
To calculate the BLEU score of the model, run the following command:
```
python3 test.py
```
The BLEU scores of all test sentences and the average BLEU score is stored in [testbleu.txt](testbleu.txt).

### Hyperparameter Tuning
To tune the hyperparameters, run the following command:
```
python3 tuning.py
```
The best model is saved at [best_model.pt](best_model.pt) and the bleu scores for all the various models are stored at [bleu_scores.pt](bleu_scores.pt).