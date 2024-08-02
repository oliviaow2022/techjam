import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW

num_labels = 3

# Define Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: torch.squeeze(value) for key, value in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

# Example data
train_texts = ["Example sentence 1", "Example sentence 2"]
train_labels = [0, 1]

test_texts = ["Test sentence 1", "Test sentence 2"]
test_labels = [0, 1]

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create datasets
train_set = TextDataset(train_texts, train_labels, tokenizer, max_length=128)
test_set = TextDataset(test_texts, test_labels, tokenizer, max_length=128)

model = BertModel.from_pretrained('bert-base-uncased')
classifier = nn.Linear(768, num_labels)
model = nn.Sequential(model, classifier)
criterion = nn.CrossEntropyLoss()  
optimizer = AdamW(model.parameters(), lr=2e-5)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
test_loader = DataLoader(test_set, batch_size=16)

def train(model, optimizer, train_loader, criterion):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Training loss: {total_loss/len(train_loader)}')

def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    total_acc = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch  
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            total_acc += (predictions == labels).sum().item()

    print(f'Test loss: {total_loss/len(test_loader)} Test acc: {total_acc/len(test_set)*100}%')

for epoch in range(3):
    train(model, optimizer, train_loader, criterion)
    evaluate(model, test_loader, criterion)

torch.save(model.state_dict(), ' sentiment_model.pt')