import torch
import torch.nn as nn
import random
from encoder import EncoderLayer, PositionalEncoding
from decoder import DecoderLayer, generate_mask
from random import shuffle

torch.manual_seed(0)

class TransformerModel(nn.Module):
    def __init__(self, in_vocab_size, out_vocab_size, d_model, num_en_layers, num_de_layers, num_heads, d_ff, max_len, dropout):
        super(TransformerModel, self).__init__()

        self.max_len = max_len

        self.encoder_embedding = nn.Embedding(in_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(out_vocab_size, d_model)

        self.pos_encoder = PositionalEncoding(d_model, max_len)

        self.encoder = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_en_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_de_layers)])

        self.fc = nn.Linear(d_model, out_vocab_size)

    def forward(self, src, tgt):
        src_mask, tgt_mask = generate_mask(src, tgt)

        src = self.encoder_embedding(src)
        tgt = self.decoder_embedding(tgt)

        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        for layer in self.encoder:
            src = layer(src, src_mask)

        for layer in self.decoder:
            tgt = layer(tgt, src, src_mask, tgt_mask)

        output = self.fc(tgt)

        return output
    
def train(model, X_train, y_train, X_val, y_val, tgt_vocab_size, optimizer, criterion, dataobj, device, epochs=10, batch_size=32):

    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        val_loss = 0
        train_steps = 0
        val_steps = 0

        combined = list(zip(X_train, y_train))
        shuffle(combined)
        X_train[:], y_train[:] = zip(*combined)


        for i in range(0, len(X_train), batch_size):

            x = X_train[i:i+batch_size]
            y = y_train[i:i+batch_size]

            src, tgt = dataobj.get_data(x, y)
            src = src.to(device)
            tgt = tgt.to(device)

            optimizer.zero_grad()

            output = model(src, tgt[:,:-1])

            output = output.contiguous().view(-1, tgt_vocab_size)
            tgt = tgt[:,1:].contiguous().view(-1)

            loss = criterion(output, tgt)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

            if train_steps % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs} - Step {train_steps} - Train Loss: {loss.item():.4f}')


        model.eval()
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):

                x = X_val[i:i+batch_size]
                y = y_val[i:i+batch_size]

                src, tgt = dataobj.get_data(x, y)
                src = src.to(device)
                tgt = tgt.to(device)

                output = model(src, tgt[:,:-1])

                output = output.contiguous().view(-1, tgt_vocab_size)
                tgt = tgt[:,1:].contiguous().view(-1)

                loss = criterion(output, tgt)

                val_loss += loss.item()
                val_steps += 1

        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/train_steps:.4f} - Val Loss: {val_loss/val_steps:.4f}')

    return model

def test_loss(model, X_test, y_test, tgt_vocab_size, criterion, dataobj, device, batch_size=32):

    model.to(device)
    model.eval()

    test_loss = 0
    test_steps = 0

    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):

            x = X_test[i:i+batch_size]
            y = y_test[i:i+batch_size]

            src, tgt = dataobj.get_data(x, y)

            src = src.to(device)
            tgt = tgt.to(device)


            output = model(src, tgt[:,:-1])

            output = output.contiguous().view(-1, tgt_vocab_size)
            tgt = tgt[:,1:].contiguous().view(-1)

            loss = criterion(output, tgt)

            test_loss += loss.item()
            test_steps += 1

    print(f'Test Loss: {test_loss/test_steps:.4f}')

    return test_loss/test_steps