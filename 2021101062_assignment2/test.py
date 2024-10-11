import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from utils import CreateDataset, inference

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

in_vocab_word2idx = np.load('in_vocab_word2idx.npy', allow_pickle=True).item()
in_vocab_idx2word = np.load('in_vocab_idx2word.npy', allow_pickle=True).item()
out_vocab_word2idx = np.load('out_vocab_word2idx.npy', allow_pickle=True).item()
out_vocab_idx2word = np.load('out_vocab_idx2word.npy', allow_pickle=True).item()

dataset = CreateDataset(in_vocab_word2idx=in_vocab_word2idx, in_vocab_idx2word=in_vocab_idx2word, out_vocab_word2idx=out_vocab_word2idx, out_vocab_idx2word=out_vocab_idx2word, is_test=True)

test_en_sentences = dataset.test_en
test_fr_sentences = dataset.test_fr

print('test size:', len(test_en_sentences))      

model = torch.load('transformer.pt')

model.to(device)

model.eval()
preds = []

for i in range(len(test_en_sentences)):
    x = test_en_sentences[i]
    y = test_fr_sentences[i]
    src,tgt = dataset.get_data([x],[y])
    src = src.to(device)

    tokens = inference(model, src, 512, device)
    pred = [out_vocab_idx2word[t] for t in tokens]
    preds.append(pred)

bleu_scores = []

with open('testbleu.txt', 'w') as f:
    for i in range(len(test_fr_sentences)):
        reference = [test_fr_sentences[i]]
        candidate = preds[i]
        score = sentence_bleu(reference, candidate, smoothing_function=SmoothingFunction().method1)
        f.write(' '.join(candidate) + '\t' + str(score) + '\n')
        bleu_scores.append(score)

print('Average BLEU score:', sum(bleu_scores)/len(bleu_scores))

with open('testbleu.txt', 'a') as f:
    f.write('Average BLEU score: ' + str(sum(bleu_scores)/len(bleu_scores)) + '\n')