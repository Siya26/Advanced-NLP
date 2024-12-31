import torch
from torch.utils.data import DataLoader
from transformers import AdamW, GPT2LMHeadModel, GPT2Tokenizer
from utils import CustomDataset, train
import time
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained('gpt2')
for param in model.parameters():
    param.requires_grad = False

for param in model.lm_head.parameters():
    param.requires_grad = True

last_layer_params = model.transformer.h[-1].parameters()
for param in last_layer_params:
    param.requires_grad = True

num_train_samples = 21000
num_val_samples = 6000

train_dataset = CustomDataset('cnn_dailymail/train.csv', tokenizer, num_train_samples)
val_dataset = CustomDataset('cnn_dailymail/validation.csv', tokenizer, num_val_samples)

batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_params}")

optimizer = AdamW([
    {'params': model.lm_head.parameters()},
    {'params': last_layer_params}
], lr=5e-5)
num_epochs = 10

model = model.to(device)

start_time = time.time()
model, train_losses, val_losses = train(model, train_loader, val_loader, optimizer, num_epochs, device, tokenizer)
print("Total time: ", time.time() - start_time, "s")

plt.plot(train_losses, label='train')
plt.plot(val_losses, label='val')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('traditional_tuning_losses.png')

torch.save(model, 'gpt2_traditional_tuning.pth')