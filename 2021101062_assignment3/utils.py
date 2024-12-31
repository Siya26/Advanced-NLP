import pandas as pd
from torch.utils.data import Dataset
import torch
from rouge import Rouge

class CustomDataset(Dataset):
    def __init__(self, data_path, tokenizer, num_samples, num_prompts=0, max_len=512):
        self.num_samples = num_samples
        self.articles, self.summaries = self.load_dataset(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_prompts = num_prompts

    def load_dataset(self, data_path):
        df = pd.read_csv(data_path)
        df = df.sample(n=self.num_samples, random_state=42).reset_index(drop=True)

        return df['article'].tolist(), df['highlights'].tolist()
    
    def get_original_summaries(self):
        return self.summaries

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx): 
        article = self.articles[idx]
        summary = self.summaries[idx]

        inputs = self.tokenizer(
            article,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            summary,
            padding='max_length',
            truncation=True,
            max_length=self.max_len + self.num_prompts,
            return_tensors='pt'
        )
        return inputs['input_ids'].squeeze(0), inputs['attention_mask'].squeeze(0), targets['input_ids'].squeeze(0)

def train(model, train_loader, val_loader, optimizer, num_epochs, device, tokenizer):
    train_losses = []
    val_losses = []
    torch.cuda.reset_max_memory_allocated(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for i, (input_ids, attention_mask, target_ids) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target_ids = target_ids.to(device)

            for j in range(target_ids.size(0)):
                target_ids[j][target_ids[j] == tokenizer.pad_token_id] = -100
            
            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_loss:.4f}")

        model.eval()
        total_loss = 0

        for i, (input_ids, attention_mask, target_ids) in enumerate(val_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target_ids = target_ids.to(device)

            for j in range(target_ids.size(0)):
                target_ids[j][target_ids[j] == tokenizer.pad_token_id] = -100

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
                loss = outputs.loss

                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        val_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_loss:.4f}")

    max_memory = torch.cuda.max_memory_allocated(device)
    print(f"Max GPU Memory Used during training: {max_memory / (1024 ** 2):.2f} MB")

    return model, train_losses, val_losses

def test(model, test_loader, device, tokenizer, original_answers):
    model.eval()

    predictions = []
    test_loss = 0

    with torch.no_grad():
        for i, (input_ids, attention_mask, target_ids) in enumerate(test_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            target_ids = target_ids.to(device)

            for j in range(target_ids.size(0)):
                target_ids[j][target_ids[j] == tokenizer.pad_token_id] = -100

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
            logits = outputs.logits
            loss = outputs.loss

            for j in range(input_ids.size(0)):
                predicted_ids = torch.argmax(logits[j], dim=-1)
                predicted_text = tokenizer.decode(predicted_ids, skip_special_tokens=True)
                predictions.append(predicted_text)

            test_loss += loss.item()

    test_loss /= len(test_loader)
    print(f"Test Loss: {test_loss}")

    rouge = Rouge()
    scores = rouge.get_scores(predictions, original_answers, avg=True)
    print("Rouge Scores: ", scores)