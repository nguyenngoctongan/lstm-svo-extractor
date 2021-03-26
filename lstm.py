import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data
from torchtext.datasets import SequenceTaggingDataset
import numpy as np
import time
import random
import pandas as pd
from torchtext.data import Dataset, Example, Field
from torchtext.data.iterator import Iterator
from torch.utils.data import DataLoader


def read_data(file_name):
    mapping = {'A3-I': 1, 'A5-B': 2, 'P-I': 3,
                'A5-I': 4, 'A1-B': 5, 'A4-B': 6,
                'A3-B': 7, 'A1-I': 8,'A4-I': 9,
                'A0-I': 10, 'A0-B': 11, 'P-B': 12,
                'A2-B': 13, 'A2-I': 14, 'O': 15}
    df = pd.read_csv(file_name, sep = '\t')
    df["label"] = df["label"].map(lambda label: mapping.get(label))
    df['word'] = df["word"].map(lambda word: str(word))
    words = list(df.head(2000)['word'])
    labels = list(df.head(2000)['label'])

    return words, labels

class BiLSTM(nn.Module):
    def __init__(self, 
                 input_dim, 
                 embedding_dim, 
                 hidden_dim, 
                 output_dim, 
                 n_layers, 
                 bidirectional, 
                 dropout, 
                 pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx = pad_idx)
        
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers = n_layers, 
                            bidirectional = bidirectional,
                            dropout = dropout if n_layers > 1 else 0)
        
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):

        embedded = self.dropout(self.embedding(text))
        outputs, (hidden, cell) = self.lstm(embedded)
        predictions = self.fc(self.dropout(outputs))
        
        return predictions

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.normal_(param.data, mean = 0, std = 0.1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def categorical_accuracy(preds, y, tag_pad_idx):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    non_pad_elements = (y != tag_pad_idx).nonzero()
    correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
    return correct.sum() / torch.FloatTensor([y[non_pad_elements].shape[0]]).to(device)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train(model, iterator, optimizer, criterion, tag_pad_idx):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        word = batch.word
        tags = batch.label
        
        optimizer.zero_grad()   
        predictions = model(word)
        
        predictions = predictions.view(-1, predictions.shape[-1])
        tags = tags.view(-1)
        
        loss = criterion(predictions, tags)      
        acc = categorical_accuracy(predictions, tags, tag_pad_idx)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion, tag_pad_idx):
    
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    
    with torch.no_grad():
        for batch in iterator:

            word = batch.word
            tags = batch.label
            
            predictions = model(word)
            predictions = predictions.view(-1, predictions.shape[-1])
            tags = tags.view(-1)
            
            loss = criterion(predictions, tags)
            acc = categorical_accuracy(predictions, tags, tag_pad_idx)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


######################################################################################################################################################
if __name__ == "__main__":

    words_train, labels_train = read_data('data/train.oie.conll')
    words_val, labels_val = read_data('data/dev.oie.conll')
    words_test, labels_test = read_data('data/test.oie.conll')


    WORD = data.Field(lower = True, tokenize=lambda x: [x])
    LABEL = data.Field(is_target=True, unk_token=None)

    fields = [("word", WORD),("label", LABEL)]

    train_examples = [data.Example.fromlist([words_train, labels_train], fields)]
    train_data = data.Dataset(train_examples, fields=fields)

    val_examples = [data.Example.fromlist([words_val, labels_val], fields)]
    val_data = data.Dataset(train_examples, fields=fields)

    test_examples = [data.Example.fromlist([words_test, labels_test], fields)]
    test_data = data.Dataset(test_examples, fields=fields)

    MAX_VOCAB_SIZE = 25000
    WORD.build_vocab(train_data, val_data,
                    max_size = MAX_VOCAB_SIZE,
                    vectors = "glove.6B.100d",
                    unk_init = torch.Tensor.normal_)
    LABEL.build_vocab(train_data,val_data)

    print(WORD.vocab.freqs.most_common(20))
    print(LABEL.vocab.itos)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    BATCH_SIZE = 3
    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, val_data, test_data), 
        batch_size = BATCH_SIZE,
        device = device)


    INPUT_DIM = len(WORD.vocab)
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 128
    OUTPUT_DIM = len(LABEL.vocab)
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.25
    PAD_IDX = WORD.vocab.stoi[WORD.pad_token]
    print(PAD_IDX)

    model = BiLSTM(INPUT_DIM, 
                            EMBEDDING_DIM, 
                            HIDDEN_DIM, 
                            OUTPUT_DIM, 
                            N_LAYERS, 
                            BIDIRECTIONAL, 
                            DROPOUT, 
                            PAD_IDX)

    model.apply(init_weights)

    print(f'The model has {count_parameters(model):,} trainable parameters')
    pretrained_embeddings = WORD.vocab.vectors

    print(pretrained_embeddings.shape)

    model.embedding.weight.data.copy_(pretrained_embeddings)

    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    print(model.embedding.weight.data)

    optimizer = optim.Adam(model.parameters())

    TAG_PAD_IDX = 0

    criterion = nn.CrossEntropyLoss(ignore_index = TAG_PAD_IDX)

    model = model.to(device)
    criterion = criterion.to(device)

    N_EPOCHS = 5

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()
        
        train_loss, train_acc = train(model, train_iterator, optimizer, criterion, TAG_PAD_IDX)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, TAG_PAD_IDX)
        
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'model.pt')
        
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    model.load_state_dict(torch.load('model.pt'))

    test_loss, test_acc = evaluate(model, test_iterator, criterion, TAG_PAD_IDX)

    print(f'Test Loss: {test_loss:.3f} |  Test Acc: {test_acc*100:.2f}%')
    