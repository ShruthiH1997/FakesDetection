#!/usr/bin/env python
# coding: utf-8

### Import data
import os
import pandas as pd
import numpy as np
import re
import glob
import tqdm
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from collections import defaultdict
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt

def clean(text):
    
    # Remove newline characters and unnecessary whitespaces
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove headers
    # text = re.sub(r'==.*?==', '', text)
    # text = re.sub(r'=.*?=', '', text)
    text = re.sub(r'=+[^=]+=+', '', text)

    return text

def preprocessing():
    # Preprocess and Save
    print("Preprocessing the data...")

    if not os.path.exists('./data_cleaned'):
        os.mkdir('./data_cleaned')

    for file in glob.glob('data/*.tok'):
        with open(file, 'r') as f:
            filename = file.split('/')[-1].replace('.tok', '')
            data = f.read().split('< start_bio >')
            pd.DataFrame([clean(para) for para in data if clean(para)]).to_csv(f'./data_cleaned/{filename}.clean', 
                                                                               index=False, header=False)

    print("Processing finished!")

def ffnn_training():
    # Train
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


    ### Training setup
    train_iter = list(all_train_data[['target', 'text']].itertuples(index=False, name=None)) 
    test_iter = list(test[['target', 'text']].itertuples(index=False, name=None)) 
    blind_iter = list(blind_test[['target', 'text']].itertuples(index=False, name=None)) 


    ### Build vocab
    tokenizer = get_tokenizer('basic_english')

    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x)


    ### Data Collator
    from torch.utils.data import DataLoader
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


    ### Model
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


    ### Training utils

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


    ### Training 

    # Hyperparameters
    EPOCHS = 10 # epoch
    LR = 0.01  # learning rate
    BATCH_SIZE = 64 # batch size for training

    criterion = torch.nn.CrossEntropyLoss()
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

    print("FFNN training started\n")
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

    print("\nFFNN training finished")

    ### Test accuracy
    print('Checking the results of test dataset.')
    accu_test, preds, labels = evaluate(test_dataloader)
    print('test accuracy {:8.3f}'.format(accu_test))
    print("confusion matrix on test data\n", confusion_matrix(labels, preds))


    ### Blind set prediction using FFNN
    blind_dataset = to_map_style_dataset(blind_iter)
    blind_dataloader = DataLoader(blind_dataset, batch_size=BATCH_SIZE,
                                 shuffle=True, collate_fn=collate_batch)

    blind_preds = predict(blind_dataloader)
    blind_prediction_save = ['REAL' if pred.item()==1 else 'FAKE' for pred in blind_preds]

    with open('blind_predictions_ffnn.txt', 'w') as f:
        for pred in blind_prediction_save:
            f.writelines(pred+'\n')

    print("Blind set predictions for FFNN saved to blind_predictions_ffnn.txt\n")


def lstm_training():
    dirpath = './data_cleaned/'
    f_train_real = list(open(dirpath+'real.train.clean','r').readlines())
    f_train_fake = list(open(dirpath+'fake.train.clean','r').readlines())
    f_valid_real = list(open(dirpath+'real.valid.clean','r').readlines())
    f_valid_fake = list(open(dirpath+'fake.valid.clean','r').readlines())
    f_test_real = list(open(dirpath+'real.test.clean','r').readlines())
    f_test_fake = list(open(dirpath+'fake.test.clean','r').readlines())
    stop_words = set(stopwords.words('english'))
    def process(string):
        # print(string)
        string = re.subn('[,"\'\;\:\(\)\n]','',string)[0]
        # print(string)
        string = re.subn('[ ]+',' ', string)[0]
        # print(string)
        string = string.strip()
        # print(string)
        string = word_tokenize(string)
        # print(string)
        string = [word for word in string if word not in stop_words]
        # print(string)
        return string
    def create_df(df_real, df_fake):
        mix_ds = []
        for line in df_real:
            ds = {}
            ds['text'] = process(line)
            ds['fake'] = 0
            mix_ds.append(ds)

        for line in df_fake:
            ds = {}
            ds['text'] = process(line)
            ds['fake'] = 1
            mix_ds.append(ds)
        return pd.DataFrame(mix_ds)
    train_set = create_df(f_train_real,f_train_fake)
    test_set = create_df(f_valid_real,f_valid_fake)
    valid_set = create_df(f_test_real,f_test_fake)

    index2word = ["<PAD>", "<SOS>", "<EOS>"]
    maxlen = 0
    for ds in [train_set, test_set, valid_set]:#, test_set, valid_set
        for txt in ds['text']:
            maxlen = max(maxlen, len(txt))
            for token in txt:
                # if token not in index2word:
                index2word.append(token)
    index2word = set(index2word)
    word2index = {token: idx for idx, token in enumerate(index2word)} 
    ##ENCODING DATA, PADDING AND TRUCNATING WHERE NEEDED
    def encode_and_pad(txt, length):
        sos = [word2index["<SOS>"]]
        eos = [word2index["<EOS>"]]
        pad = [word2index["<PAD>"]]

        if len(txt) < length - 2: # -2 for SOS and EOS
            n_pads = length - 2 - len(txt)
            encoded = [word2index[w] for w in txt]
            return sos + encoded + eos + pad * n_pads 
        else: # txt is longer than possible; truncating
            encoded = [word2index[w] for w in txt]
            truncated = encoded[:length - 2]
            return sos + truncated + eos
    
    seq_length = 256
    batch_size = 50
    ##ENCODING DATA
    train_encoded = [(encode_and_pad(row['text'], seq_length), row['fake']) for x, row in train_set.iterrows()]
    valid_encoded = [(encode_and_pad(row['text'], seq_length), row['fake']) for x, row in valid_set.iterrows()]
    test_encoded = [(encode_and_pad(row['text'], seq_length), row['fake']) for x, row in test_set.iterrows()]
    ##CONVERTING DATA FOR TORCH DATALOADER INPUT
    train_x = np.array([text for text, label in train_encoded])
    train_y = np.array([label for text, label in train_encoded])
    valid_x = np.array([text for text, label in valid_encoded])
    valid_y = np.array([label for text, label in valid_encoded])
    test_x = np.array([text for text, label in test_encoded])
    test_y = np.array([label for text, label in test_encoded])

    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_ds = TensorDataset(torch.from_numpy(valid_x), torch.from_numpy(valid_y))
    test_ds = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
    ##CREATING DATALOADER
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, drop_last=True)
    val_dl = DataLoader(valid_ds, shuffle=True, batch_size=batch_size, drop_last=True)
    test_dl = DataLoader(test_ds, shuffle=True, batch_size=batch_size, drop_last=True)    
    ##DEFINING MODEL
    class BiLSTM_FakeDetection(torch.nn.Module) :
        def __init__(self, vocab_size, embedding_dim, hidden_dim, dropout) :
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional = False)
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden_dim, 2)
        ##FORWARD PASS
        def forward(self, x, hidden):
            embs = self.embedding(x)
            out, hidden = self.lstm(embs, hidden)
            out = self.dropout(out)
            out = self.fc2(out)
            out = out[:, -1]
            return out, hidden
        
        def init_hidden(self):
            # return (torch.zeros(2, batch_size, 64), torch.zeros(2, batch_size, 64))
            return (torch.zeros(1, batch_size, 64), torch.zeros(1, batch_size, 64))
    ##CREATING MODEL OBJECT AND DEFINING HYPERPARAMERTERS
    model = BiLSTM_FakeDetection(len(word2index), 128, 64, 0.2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 3e-4)
    epochs = 50
    ##INITIALING HIDDEN STATE AND CELL STATE
    h0, c0 =  model.init_hidden()
    h0 = h0.to(device)
    c0 = c0.to(device)
    ##EPOCH TRAINING FUNCTION
    def train_epoch(model, data_loader,optimizer,device,n_examples):
        model = model.train()
        batch_acc = []
        losses = []
        correct_predictions = 0

        for batch_idx, batch in enumerate(data_loader):

            input = batch[0].to(device)
            target = batch[1].to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                out, hidden = model(input, (h0, c0))
                _, preds = torch.max(out, 1)
                preds = preds.to(device).tolist()
                batch_acc.append(accuracy_score(preds, target.tolist()))
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
        return sum(batch_acc)*1.0/len(batch_acc), np.mean(losses)
    ##EVALUATION FUNCTION
    def eval_model(model, data_loader, loss_fn, device, n_examples):
        batch_acc = []
        losses = []
        for batch_idx, batch in enumerate(test_dl):

            input = batch[0].to(device)
            target = batch[1].to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                out, hidden = model(input, (h0, c0))
                loss = criterion(out, target)
                _, preds = torch.max(out, 1)
                preds = preds.to(device).tolist()
                batch_acc.append(accuracy_score(preds, target.tolist()))
            losses.append(loss.item())

        return sum(batch_acc)*1.0/len(batch_acc), np.mean(losses)
    ##RUNNING FOR ALL EPOCHS
    history = defaultdict(list)
    best_accuracy = 0
    loss_fn = criterion
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(
            model,
            train_dl,    
            # loss_fn, 
            optimizer, 
            device, 
            len(train_x)
        )

        print(f'Train loss {train_loss} accuracy {train_acc}')

        val_acc, val_loss = eval_model(
            model,
            val_dl,
            loss_fn, 
            device, 
            len(valid_x)
        )

        print(f'Val loss {val_loss} accuracy {val_acc}')
        print()

        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            h_final, c_final = h0, c0
            best_accuracy = val_acc
    ##PLOTTING LOSS CURVES
    plt.plot(history['train_loss'], label = 'training_loss')
    plt.plot(history['val_loss'], label = 'validation_loss')
    plt.legend(loc="upper left")
    plt.title("loss vs epochs")
    plt.show()
    ##PLOTTING TRAINING CURVES
    plt.plot(history['train_acc'], label = 'training_acc')
    plt.plot(history['val_acc'], label = 'validation_acc')
    plt.legend(loc="upper left")
    plt.title("accuracy vs epochs")
    ##TESTING ON BEST MODEL VERSION
    model = BiLSTM_FakeDetection(len(word2index), 128, 64, 0.2)
    model.load_state_dict(torch.load('best_model_state.bin'))
    model = model.to(device) 

    test_acc, _ = val_acc, val_loss = eval_model(
    model,
    test_dl,
    loss_fn, 
    device, 
    len(test_x)
  )

    print("\n Test accuracy: ",test_acc.item(), "\n")
    ##CONFUSION MATRIX FUNCTION
    def gen_confusion_matrix(model, data_loader, loss_fn, device, n_examples):
        total_preds = []
        total_actual = []
        losses = []
        
        for batch_idx, batch in enumerate(test_dl):

            input = batch[0].to(device)
            target = batch[1].to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(False):
                out, hidden = model(input, (h0, c0))
                _, preds = torch.max(out, 1)
                total_preds.extend(preds.to(device).tolist())
                total_actual.extend(target.tolist())

        return confusion_matrix(total_preds,total_actual)
    disp = ConfusionMatrixDisplay(confusion_matrix=gen_confusion_matrix(
    model,
    test_dl,
    loss_fn, 
    device, 
    len(test_x)
  ),display_labels=['Real','Fake'])
    disp.plot()


####################### Preprocessing the data #######################
print("Make sure the data is present in './data")
preprocessing()
    
####################### Training the FFNN #######################
ffnn_training()
####################### Training the LSTM #######################

lstm_training()

