import torch
import torch.nn as nn
import torch.optim as optim

from utils import CreateDataset
from transformer import TransformerModel, train, test_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = CreateDataset()

train_en_sentences = dataset.train_en
train_fr_sentences = dataset.train_fr
val_en_sentences = dataset.val_en
val_fr_sentences = dataset.val_fr
test_en_sentences = dataset.test_en
test_fr_sentences = dataset.test_fr

print('train size:', len(train_en_sentences))
print('val size:', len(val_en_sentences))
print('test size:', len(test_en_sentences))

in_vocab_size = len(dataset.in_vocab_word2idx)
out_vocab_size = len(dataset.out_vocab_word2idx)
d_model = 512
n_heads = 8
n_en_layers = 1
n_de_layers = 1
d_ff = 1024
dropout = 0.1
lr = 0.001
epochs = 5
batch_size = 64
max_len = 1000

print('training started')

model = TransformerModel(in_vocab_size, out_vocab_size, d_model, n_en_layers, n_de_layers, n_heads, d_ff, max_len, dropout)

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss().to(device)

trained_model = train(model, train_en_sentences, train_fr_sentences, val_en_sentences, val_fr_sentences, out_vocab_size, optimizer, criterion, dataset, device, epochs, batch_size)

torch.save(trained_model, 'transformer.pt')

test_loss(trained_model, test_en_sentences, test_fr_sentences, out_vocab_size, criterion, dataset, device, batch_size)