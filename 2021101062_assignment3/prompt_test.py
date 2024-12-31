import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from utils import CustomDataset, test
from prompt_tuning import GPT2WithSoftPrompts, SoftPromptEmbedding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

prompt = '[SUMMARIZE]'
num_prompts = len(tokenizer.encode(prompt, add_special_tokens=False))

num_test_samples = 3000
test_dataset = CustomDataset('cnn_dailymail/test.csv', tokenizer, num_prompts, num_test_samples)

original_summaries = test_dataset.get_original_summaries()

batch_size = 8
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = torch.load('gpt2_soft_prompts.pth')
model = model.to(device)

test(model, test_loader, device, tokenizer, original_summaries)