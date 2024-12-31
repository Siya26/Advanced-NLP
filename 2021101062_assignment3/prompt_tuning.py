import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, GPT2LMHeadModel, GPT2Tokenizer
from utils import CustomDataset, train
import time
import matplotlib.pyplot as plt
    
class SoftPromptEmbedding(nn.Module):
    def __init__(self, model, num_prompts, embedding_size, tokenizer, prompt):
        super().__init__()
        self.soft_prompts = nn.Embedding(num_prompts, embedding_size)
        
        token_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
        token_embeddings = model.transformer.wte(token_ids).squeeze(0)
        
        with torch.no_grad():
            self.soft_prompts.weight.copy_(token_embeddings)

    def forward(self, batch_size):
        soft_prompts = self.soft_prompts.weight.unsqueeze(0).expand(batch_size, -1, -1)
        return soft_prompts

class GPT2WithSoftPrompts(nn.Module):
    def __init__(self, model, num_prompts, tokenizer, prompt, embedding_size=768):
        super().__init__()
        self.model = model
        self.soft_prompt_embeddings = SoftPromptEmbedding(model, num_prompts, embedding_size, tokenizer, prompt)

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size = input_ids.size(0)
        
        soft_prompts = self.soft_prompt_embeddings(batch_size)
        input_embeddings = self.model.transformer.wte(input_ids)
        
        inputs_embeds = torch.cat([soft_prompts, input_embeddings], dim=1)
        
        if attention_mask is not None:
            soft_attention_mask = torch.ones((batch_size, soft_prompts.size(1)), device=attention_mask.device)
            attention_mask = torch.cat([soft_attention_mask, attention_mask], dim=1)
        
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return outputs
    
if __name__ == '__main__':  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    for param in model.parameters():
        param.requires_grad = False  
        
    prompt = '[SUMMARIZE]'
    num_prompts = len(tokenizer.encode(prompt, add_special_tokens=False))

    num_train_samples = 21000
    num_val_samples = 6000

    train_dataset = CustomDataset('cnn_dailymail/train.csv', tokenizer, num_train_samples, num_prompts)
    val_dataset = CustomDataset('cnn_dailymail/validation.csv', tokenizer, num_val_samples, num_prompts)

    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = GPT2WithSoftPrompts(model, num_prompts, tokenizer, prompt).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")

    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_epochs = 10

    start_time = time.time()
    model, train_losses, val_losses = train(model, train_loader, val_loader, optimizer, num_epochs, device, tokenizer)
    print("Total time: ", time.time() - start_time, "s")

    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('prompt_tuning_losses.png')

    torch.save(model, 'gpt2_soft_prompts.pth')