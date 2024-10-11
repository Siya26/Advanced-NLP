import numpy as np
import torch
import torch.nn as nn
import nltk
import re

corpus = open('Auguste_Maquet.txt', 'r').read()
sentences = nltk.sent_tokenize(corpus)

embeddings = {}
with open('glove.6B.100d.txt', 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings[word] = coefs

class Dataset:
    def __init__(self, sentences, embeddings, train=False, word2idx=None):
        self.sentences = sentences
        self.embeddings = embeddings
        self.word2idx = word2idx
        self.train = train
        self.words = None

        self.preprocess()
        self.add_extra_tokens()
        if word2idx is None:
            self.create_vocab()        
        self.get_embeddings()

    def preprocess(self):
        processed_sentences = []
        for sentence in self.sentences:
            sentence = re.sub(r'[^\w\s]', '', sentence)  
            sentence = sentence.replace('_', ' ')  
            sentence = sentence.replace('-', ' ')  
            sentence = sentence.replace('\n', ' ')  
            sentence = sentence.strip()
            processed_sentences.append(sentence)

        self.sentences = [sentence.lower() for sentence in processed_sentences if len(sentence.split()) > 5]
        self.words = [nltk.word_tokenize(sentence) for sentence in self.sentences]

    def add_extra_tokens(self):
        for i in range(len(self.words)):
            for j in range(len(self.words[i])):
                if self.train:
                    if self.words[i][j] not in self.embeddings:
                        self.words[i][j] = '<unk>'
                elif self.words[i][j] not in self.word2idx:
                    self.words[i][j] = '<unk>'
                
            self.words[i] = ['<s>'] + self.words[i] + ['</s>']

    def create_vocab(self):
        self.word2idx = {}
        i = 0

        for sentence in self.words:
            for word in sentence:
                if word not in self.word2idx:
                    self.word2idx[word] = i
                    i += 1

        self.word2idx['<pad>'] = i

    def get_embeddings(self):
        embeddings['<unk>'] = np.zeros(100)
        embeddings['<s>'] = np.ones(100)
        embeddings['</s>'] = np.ones(100)
        embeddings['<pad>'] = np.ones(100)

    def create_data(self, sentences):
        self.X = []
        self.y = []

        max_len = max([len(sentence) for sentence in sentences])
        for i in range(len(sentences)):
            sentences[i] = sentences[i] + ['<pad>']*(max_len - len(sentences[i]))

        for sentence in sentences:
            X = [self.embeddings[sentence[j]] for j in range(0, len(sentence) - 1)]
            y_temp = [self.word2idx[sentence[j]] for j in range(1, len(sentence))]
            y = np.zeros((len(y_temp), len(self.word2idx)))
            for i in range(len(y_temp)):
                y[i][y_temp[i]] = 1

            self.X.append(X)
            self.y.append(y)

        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.X = torch.tensor(self.X).float()
        self.y = torch.tensor(self.y).float()

np.random.shuffle(sentences)
train_sentences = sentences[:int(0.7 * len(sentences))]
val_sentences = sentences[int(0.7 * len(sentences)):int(0.9 * len(sentences))]
test_sentences = sentences[int(0.9 * len(sentences)):]

train_dataset = Dataset(train_sentences, embeddings, train=True)
print("Train dataset created")
val_dataset = Dataset(val_sentences, embeddings, word2idx=train_dataset.word2idx)
print("Val dataset created")
test_dataset = Dataset(test_sentences, embeddings, word2idx=train_dataset.word2idx)
print("Test dataset created")

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h_0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        c_0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        r_out, _ = self.lstm(x, (h_0, c_0))
        output = self.fc(r_out)
        return output

input_dim = 100
hidden_dim = 300
output_dim = len(train_dataset.word2idx)

num_epochs = 10
batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

model = LSTM(input_dim, hidden_dim, output_dim)
model.to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_steps = 0

    for i in range(0, len(train_dataset.words), batch_size):
        temp_train_data = train_dataset.words[i:i+batch_size]
        train_dataset.create_data(temp_train_data)
        X = train_dataset.X.to(device).float()
        y = train_dataset.y.to(device).float()
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output.view(-1, output_dim), y.view(-1, output_dim))
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_steps += 1

    train_loss /= train_steps
    print(f'Epoch: {epoch + 1}, Training Loss: {train_loss}')

    model.eval()
    val_loss = 0
    val_steps = 0

    with torch.no_grad():
        for i in range(0, len(val_dataset.words), batch_size):
            temp_val_data = val_dataset.words[i:i+batch_size]
            val_dataset.create_data(temp_val_data)
            X = val_dataset.X.to(device).float()
            y = val_dataset.y.to(device).float()
            output = model(X)
            loss = criterion(output.view(-1, output_dim), y.view(-1, output_dim))
            val_loss += loss.item()
            val_steps += 1

    val_loss /= val_steps
    print(f'Epoch: {epoch + 1}, Validation Loss: {val_loss}')

torch.save(model.state_dict(), 'LSTM.pt')

# test
model.eval()
test_loss = 0
test_steps = 0

with torch.no_grad():
    for i in range(0, len(test_dataset.words), batch_size):
        temp_test_data = test_dataset.words[i:i+batch_size]
        test_dataset.create_data(temp_test_data)
        X = test_dataset.X.to(device).float()
        y = test_dataset.y.to(device).float()
        output = model(X)
        loss = criterion(output.view(-1, output_dim), y.view(-1, output_dim))
        test_loss += loss.item()
        test_steps += 1

test_loss /= test_steps
print(f'Test Loss: {test_loss}')

def perplexity(model, dataset, file_name):
    model = model.to(device)
    total_perplexity = 0
    total = 0
    model.eval()

    with open(file_name, 'a') as f:
        with torch.no_grad():
            for sentence, words in zip(dataset.sentences, dataset.words):
                dataset.create_data([words])
                if len(dataset.X) == 0:
                    continue

                X = dataset.X.to(device).float()
                y = dataset.y.to(device).float()
                output = model(X)
                loss = criterion(output.view(-1, output_dim), y.view(-1, output_dim))
                perplexity = torch.exp(loss).item()
                total_perplexity += perplexity
                total += 1

                f.write(f'{sentence}\t{perplexity}\n')

            total_perplexity /= total
            f.write(f'\nAverage Perplexity: {total_perplexity}')

perplexity(model, train_dataset, '2021101062-LM2-train-perplexity.txt')
perplexity(model, val_dataset, '2021101062-LM2-val-perplexity.txt')
perplexity(model, test_dataset, '2021101062-LM2-test-perplexity.txt')