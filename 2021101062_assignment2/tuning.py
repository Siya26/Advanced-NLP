import torch
import torch.nn as nn
import torch.optim as optim

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from utils import CreateDataset, inference
from transformer import TransformerModel, train

dataset = CreateDataset()

train_en_sentences = dataset.train_en
train_fr_sentences = dataset.train_fr
val_en_sentences = dataset.val_en
val_fr_sentences = dataset.val_fr
test_en_sentences = dataset.test_en
test_fr_sentences = dataset.test_fr

in_vocab_size = len(dataset.in_vocab_word2idx)
out_vocab_size = len(dataset.out_vocab_word2idx)

d_ff = 2048
lr = 0.001
epochs = 5
batch_size = 64
max_len = 1000
d_model = [256, 512]
n_heads = [4, 8,]
n_en_layers = [1, 2]
n_de_layers = [1, 2]
dropout = [0.1, 0.2]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

best_bleu = 0
best_hyperparameters = None

for d in d_model:
    for h in n_heads:
        for en in n_en_layers:
            for de in n_de_layers:
                for do in dropout:

                    model = TransformerModel(in_vocab_size, out_vocab_size, d, en, de, h, d_ff, max_len, do)
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    criterion = nn.CrossEntropyLoss().to(device)

                    trained_model = train(model, train_en_sentences, train_fr_sentences, val_en_sentences, val_fr_sentences, out_vocab_size, optimizer, criterion, dataset, device, epochs, batch_size)
                    trained_model.eval()

                    preds = []

                    for i in range(len(test_en_sentences)):
                        x = test_en_sentences[i]
                        y = test_fr_sentences[i]
                        src,tgt = dataset.get_data([x],[y])
                        src = src.to(device)

                        tokens = inference(trained_model, src, 512, device)
                        pred = [dataset.out_vocab_idx2word[t] for t in tokens]
                        preds.append(pred)

                    bleu_scores = []
                    for i in range(len(test_fr_sentences)):
                        reference = [test_fr_sentences[i]]
                        candidate = preds[i]
                        score = sentence_bleu(reference, candidate, smoothing_function=SmoothingFunction().method1)
                        bleu_scores.append(score)

                    bleu = sum(bleu_scores)/len(bleu_scores)

                    print('--------------------------------------------------------------------------------------------------------')
                    print('\n')
                    print('d_model:', d, 'n_heads:', h, 'n_en_layers:', en, 'n_de_layers:', de, 'dropout:', do, 'BLEU:', bleu)
                    print('\n')
                    print('--------------------------------------------------------------------------------------------------------')

                    if bleu > best_bleu:
                        best_bleu = bleu
                        best_hyperparameters = (d, h, en, de, do)
                        torch.save(trained_model, 'transformer.pt')

print('Best hyperparameters:', best_hyperparameters)
print('Best BLEU:', best_bleu)