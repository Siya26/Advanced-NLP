import re
import spacy
import numpy as np
import torch
from torch.utils.data import Dataset

class CreateDataset(Dataset):
    def __init__(self, cut_off_freq = 2, in_vocab_word2idx = None, out_vocab_word2idx = None, out_vocab_idx2word = None, is_test = False):
        self.nlp_en = spacy.load("en_core_web_sm")
        self.nlp_fr = spacy.load("fr_core_news_sm")

        if is_test:
            self.in_vocab_word2idx = in_vocab_word2idx
            self.out_vocab_word2idx = out_vocab_word2idx
            self.out_vocab_idx2word = out_vocab_idx2word
        else:
            self.in_vocab_word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
            self.out_vocab_word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
            self.out_vocab_idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>'}

        if not is_test:
            self.cut_off_freq = cut_off_freq
            self.train_en, self.train_fr = self.get_tokens('ted-talks-corpus/train.en', 'ted-talks-corpus/train.fr')
            self.val_en, self.val_fr = self.get_tokens('ted-talks-corpus/dev.en', 'ted-talks-corpus/dev.fr')
        self.test_en, self.test_fr = self.get_tokens('ted-talks-corpus/test.en', 'ted-talks-corpus/test.fr')

        print('tokenization done')

        if not is_test:
            self.train_en = self.add_unks_train(self.train_en)
            self.train_fr = self.add_unks_train(self.train_fr)
            self.create_vocab()
            np.save('in_vocab_word2idx.npy', self.in_vocab_word2idx)
            np.save('out_vocab_word2idx.npy', self.out_vocab_word2idx)
            np.save('out_vocab_idx2word.npy', self.out_vocab_idx2word)

        print('vocab created')

        if not is_test:
            self.val_en = self.add_unks_test(self.val_en, 'en')
            self.val_fr = self.add_unks_test(self.val_fr, 'fr')
            self.train_en = self.add_start_end(self.train_en)
            self.train_fr = self.add_start_end(self.train_fr)
            self.val_en = self.add_start_end(self.val_en)
            self.val_fr = self.add_start_end(self.val_fr)
        self.test_en = self.add_unks_test(self.test_en, 'en')
        self.test_fr = self.add_unks_test(self.test_fr, 'fr')
        self.test_en = self.add_start_end(self.test_en)
        self.test_fr = self.add_start_end(self.test_fr)

    def preprocess(self, text):
        text = re.sub(' +', ' ', text)
        text = text.lower()
        text = text.strip()
        return text
    
    def tokenize(self, text, lang):
        if lang == 'en':
            doc = self.nlp_en(text)
        else:
            doc = self.nlp_fr(text)
        return [token.text for token in doc]
    
    def read_file(self, file_en, file_fr):
        en = open(file_en, 'r').read().strip().split('\n')
        fr = open(file_fr, 'r').read().strip().split('\n')
        return en, fr
    
    def get_tokens(self, file_en, file_fr):
        en, fr = self.read_file(file_en, file_fr)
        en = [self.preprocess(text) for text in en]
        fr = [self.preprocess(text) for text in fr]
        en_tokens = [self.tokenize(text, 'en') for text in en]
        fr_tokens = [self.tokenize(text, 'fr') for text in fr]
        return en_tokens, fr_tokens
    
    def add_unks_train(self, tokens):
        word_freq = {}
        for sentence in tokens:
            for word in sentence:
                if word not in word_freq:
                    word_freq[word] = 1
                else:
                    word_freq[word] += 1
        for sentence in tokens:
            for word in sentence:
                if word_freq[word] < self.cut_off_freq:
                    word = '<UNK>'
        return tokens
    
    def create_vocab(self):
        for sentence in self.train_en:
            for word in sentence:
                if word not in self.in_vocab_word2idx:
                    idx = len(self.in_vocab_word2idx)
                    self.in_vocab_word2idx[word] = idx
        for sentence in self.train_fr:
            for word in sentence:
                if word not in self.out_vocab_word2idx:
                    idx = len(self.out_vocab_word2idx)
                    self.out_vocab_word2idx[word] = idx
                    self.out_vocab_idx2word[idx] = word

    def add_unks_test(self, tokens, lang):
        if lang == 'en':
            vocab = self.in_vocab_word2idx
        else:
            vocab = self.out_vocab_word2idx

        for sentence in tokens:
            for i in range(len(sentence)):
                if sentence[i] not in vocab:
                    sentence[i] = '<UNK>'
        return tokens
    
    def add_start_end(self, tokens):
        for i in range(len(tokens)):
            tokens[i] = ['<SOS>'] + tokens[i] + ['<EOS>']
        return tokens
    
    def get_data(self, en_sentences, fr_sentences):
        max_len_en = max([len(sentence) for sentence in en_sentences])
        max_len_fr = max([len(sentence) for sentence in fr_sentences])

        for i in range(len(en_sentences)):
            en_sentences[i] += ['<PAD>'] * (max_len_en - len(en_sentences[i]))
            fr_sentences[i] += ['<PAD>'] * (max_len_fr - len(fr_sentences[i]))

        X = [[self.in_vocab_word2idx[word] for word in sentence] for sentence in en_sentences]
        Y = [[self.out_vocab_word2idx[word] for word in sentence] for sentence in fr_sentences]
        
        return torch.tensor(X), torch.tensor(Y)
    
def inference(model, src, max_len, device):
        src = model.encoder_embedding(src)
        src = model.pos_encoder(src)

        for layer in model.encoder:
            src = layer(src)

        tgt = torch.tensor([2], dtype=torch.long).unsqueeze(0)
        tgt = tgt.to(device)

        pred_tokens = []

        for i in range(max_len):
            tgt1 = model.decoder_embedding(tgt)
            tgt1 = model.pos_encoder(tgt1)

            for layer in model.decoder:
                tgt1 = layer(tgt1, src)

            output = model.fc(tgt1)
            output = output.argmax(dim=-1)
            pred_tokens.append(output[0, -1].item())

            new_token = output[0, -1].unsqueeze(0).unsqueeze(0)
            tgt = torch.cat((tgt, new_token), dim=1)
            if output[0, -1] == 3:
                break

        return pred_tokens