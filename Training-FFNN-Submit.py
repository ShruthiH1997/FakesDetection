import os
import pandas as pd
import numpy as np
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
import time
from sklearn.metrics import confusion_matrix

real_train = pd.read_table('./data_cleaned/real.train.clean', header=None, names=['text'])
real_train['target'] = 1
fake_train = pd.read_table('./data_cleaned/fake.train.clean', header=None, names=['text'])
fake_train['target'] = 0
train = pd.concat([real_train, fake_train], axis=0)

# Valid
real_valid = pd.read_table('./data_cleaned/real.valid.clean', header=None, names=['text'])
real_valid['target'] = 1
fake_valid = pd.read_table('./data_cleaned/fake.valid.clean', header=None, names=['text'])
fake_valid['target'] = 0
valid = pd.concat([real_valid, fake_valid], axis=0)

# Test
real_test = pd.read_table('./data_cleaned/real.test.clean', header=None, names=['text'])
real_test['target'] = 1
fake_test = pd.read_table('./data_cleaned/fake.test.clean', header=None, names=['text'])
fake_test['target'] = 0
test = pd.concat([real_test, fake_test], axis=0)

# Blind test
blind_test = pd.read_table('./data_cleaned/blind.test.clean', header=None, names=['text'])
blind_test['target'] = -999

all_train_data = pd.concat([train, valid]).reset_index(drop=True)

train_iter = list(all_train_data[['target', 'text']].itertuples(index=False, name=None)) 
test_iter = list(test[['target', 'text']].itertuples(index=False, name=None)) 
blind_iter = list(blind_test[['target', 'text']].itertuples(index=False, name=None)) 

tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)
        
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])


text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: int(x)


# ### Data Collator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)


# ### Model

# In[13]:


from torch import nn
import torch.nn.functional as F
import torch.optim as optim


# In[14]:


# Vanilla deeper

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False) # set sparse to True if using SGD, False for Adam
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        out = self.fc1(embedded)
        out = self.fc2(out)
        return out
    

num_class = len(set([label for (label, text) in train_iter]))
vocab_size = len(vocab)
emsize = 64
hidden_dim = 32
model = TextClassificationModel(vocab_size, emsize, hidden_dim, num_class).to(device)


def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predicted_label = model(text, offsets)
        loss = criterion(predicted_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader):
    batch_size = dataloader.batch_size
    length = dataloader.__len__()
    model.eval()
    total_acc, total_count = 0, 0
    predictions = []
    labels = []
    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            predictions.extend(predicted_label.argmax(1))
            labels.extend(label)
            # predictions[idx, :predicted_label.shape[0]] = predicted_label.argmax(1)
            # labels[idx, :label.shape[0]] = label
            total_count += label.size(0)
    return total_acc/total_count, predictions, labels

def predict(dataloader):
    batch_size = dataloader.batch_size
    length = dataloader.__len__()
    model.eval()
    total_acc, total_count = 0, 0
    # predictions = torch.empty(length, batch_size)
    predictions = []
    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predicted_label = model(text, offsets)
            # predictions[idx, :predicted_label.shape[0]] = predicted_label.argmax(1)
            predictions.extend(predicted_label.argmax(1))
    return predictions


# ### Training 

# In[19]:


from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset

# Hyperparameters
EPOCHS = 10 # epoch
LR = 0.01  # learning rate
BATCH_SIZE = 64 # batch size for training

criterion = torch.nn.CrossEntropyLoss()
# criterion = torch.nn.BCELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=LR)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None
train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)
num_train = int(len(train_dataset) * 0.95)
split_train_, split_valid_ = \
    random_split(train_dataset, [num_train, len(train_dataset) - num_train])

train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_batch)


# In[22]:


for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val, preds_val, labels = evaluate(valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
      scheduler.step()
    else:
       total_accu = accu_val
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val))
    print('-' * 59)


print('Checking the results of test dataset.')
accu_test, preds, labels = evaluate(test_dataloader)
print('test accuracy {:8.3f}'.format(accu_test))


# In[25]:


confusion_matrix(labels, preds)

blind_dataset = to_map_style_dataset(blind_iter)
blind_dataloader = DataLoader(blind_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_batch)


blind_preds = predict(blind_dataloader)

blind_prediction_save = ['REAL' if pred.item()==1 else 'FAKE' for pred in blind_preds]


with open('blind_predictions_ffnn.txt', 'w') as f:
    for pred in blind_prediction_save:
        f.writelines(pred+'\n')





