#!/usr/bin/env python
# coding: utf-8

# In[11]:


import torch
import pandas as pd
from torchtext import data
from torchtext.vocab import Vectors
from torch.nn import init
from tqdm import tqdm
from torchtext.vocab import GloVe
from torchtext import data


# In[12]:


torch.cuda.is_available()


# # data 

# ### my dataset 

# In[13]:


def get_dataset(csv_data, id_field, text_field, label_field, is_final_valid = False):
    fields = [('id', id_field), ('new', text_field), ('original2', text_field), ('meanGrade', label_field)]
    fields2 = [('id', id_field), ('new', text_field), ('original2', text_field)]
    examples = []
    if is_final_valid:
        for myid, new, original2 in tqdm(zip(csv_data['id'], csv_data['new'],csv_data['original2'])):
            examples.append(data.Example.fromlist([myid, new, original2], fields2))
        return examples, fields2
    else:
        for myid, new, original2, label in tqdm(zip(csv_data['id'], csv_data['new'], csv_data['original2'], csv_data['meanGrade'])):
            examples.append(data.Example.fromlist([myid, new, original2, label], fields))
        return examples, fields


# ### load data

# In[14]:


#tokenize = lambda x: x.split()
TEXT = data.RawField()
LABEL = data.LabelField(use_vocab=False, dtype=torch.float)
ID = data.LabelField(use_vocab=False)

train_path = "../data/task-1/train2.csv"
valid_path = "../data/task-1/dev2.csv"
test_path = "../data/task-1/test2.csv"
    
train = pd.read_csv(train_path)
valid = pd.read_csv(valid_path)
test = pd.read_csv(test_path)

test = test.reset_index(drop=True)
valid = valid.reset_index(drop=True)


# ### split data 

# In[15]:


train_examples, train_fields = get_dataset(train, ID, TEXT, LABEL)
valid_examples, valid_fields = get_dataset(valid, ID, TEXT, LABEL)
#test_examples, test_fields = get_dataset(test, TEXT, None, True)
test_examples, test_fields = get_dataset(test, ID, TEXT, LABEL)
#final_valid_examples, final_valid_fields = get_dataset(final_valid, ID, TEXT, LABEL, True)


train_data = data.Dataset(train_examples, train_fields)
valid_data = data.Dataset(valid_examples, valid_fields)
test_data = data.Dataset(test_examples, test_fields)
#final_valid_data = data.Dataset(final_valid_examples, final_valid_fields)

print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')
#print(f'Number of final_valid examples: {len(final_valid_data)}')


# ### import Glove

# In[16]:


# TEXT.build_vocab(train_data,vectors="glove.840B.300d", unk_init=torch.Tensor.normal_) 
# TEXT.build_vocab(train_data) 
LABEL.build_vocab(train_data)
# print(f"Unique tokens in TEXT vocabulary: {len(TEXT.vocab)}")
print(f"Unique tokens in LABEL vocabulary: {len(LABEL.vocab)}")


# ### baches example

# In[18]:


BATCH_SIZE = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data), 
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.new), # the BucketIterator needs to be told what function it should use to group the data.
    sort_within_batch=False,
    device=device)


# # Elmo_BiLSTM 

# In[19]:


import torch.nn as nn
import torch.nn.functional as F

class BiLSTMModel(nn.Module):
    def __init__(self, embedding_dim, output_dim, hidden_dim, dropout):
        super().__init__()
        #self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True) 
        self.fc = nn.Linear(EMBEDDING_DIM*4, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q2, q3):
        # embedding[sent_len, betch_size, embedding_dim]
        
        # text[sent_len, batch_size]
        # embedded = self.embedding()
        # embeded[sent_len, batch_size, embedding_dim]
        m = [q2, q3, (q2-q3).abs(), q2*q3]
        pair_emb = torch.cat(m, dim=-1)
        #print(q2.size())
        #print(pair_emb.size())
        # hidden[batch_size, hidden_dim * num_directions]
        return self.fc(pair_emb)


# # training

# ## parameters

# In[20]:


EMBEDDING_DIM = 1024
OUTPUT_DIM = 1 # Classification: num_labels/Regression: 1
HIDDEN_SIZE = 16 
DROPOUT = 0.5 

model = BiLSTMModel(EMBEDDING_DIM, OUTPUT_DIM, HIDDEN_SIZE, DROPOUT)


# In[21]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')


# ## training

# In[22]:


import torch.optim as optim

optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss(reduction='sum') # TODO
model = model.to(device)
criterion = criterion.to(device)


# ### Elmo

# In[36]:


from allennlp.modules.elmo import Elmo, batch_to_ids
options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 2, dropout=0)

def get_elmo_embeddings(batch):
    sentences = [sen.split()for sen in batch]
    length = torch.Tensor([len(sentences[i]) for i in range(len(sentences))])
    character_ids = batch_to_ids(sentences)
    embeddings = elmo(character_ids)
    #embedding = embeddings['elmo_representations'][0].permute(1,2,0).max(dim=0)[0]
    embedding = embeddings['elmo_representations'][0].permute(1,2,0)[0]
    embedding = torch.div(embedding, length).permute(1,0)
    ### embedding [sent_len, batch_size, embed_dim] -> embedding [batch_size, embed_dim]
    return embedding
    


# In[37]:


def trainModel(model, iterator, optimizer, criterion):
    epoch_loss = 0
    model.train()
    
    for batch in iterator:
        optimizer.zero_grad()
        e1 = get_elmo_embeddings(batch.new).to(device)
        e2 = get_elmo_embeddings(batch.original2).to(device)
        predictions = model(e1,e2).squeeze(1)
        loss = criterion(predictions, batch.meanGrade)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        
    return epoch_loss / len(iterator)
        


# In[38]:


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            e1 = get_elmo_embeddings(batch.new).to(device)
            e2 = get_elmo_embeddings(batch.original2).to(device)
            predictions = model(e1,e2).squeeze(1)
            #predictions = model(embeddings).squeeze(1)
            loss = criterion(predictions, batch.meanGrade)
        
            epoch_loss += loss.item()
            
    return epoch_loss / len(iterator)


# In[ ]:


N_EPOCHS = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    trainModel(model, train_iterator, optimizer, criterion)
    
    train_loss = trainModel(model, train_iterator, optimizer, criterion)
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'lstm-model.pth')
        print("save")
    
    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\tValid Loss: {valid_loss:.3f}')
    


# # Evaluation (MSE)

# In[26]:


import numpy as np
def myMSE(model, iterator):
    pred_list = np.array([])
    real_list = np.array([])
    id_list = np.array([])
    
#     Result_Dict = sorted(list(set(train['meanGrade'])))
    
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            e1 = get_elmo_embeddings(batch.new).to(device)
            e2 = get_elmo_embeddings(batch.original2).to(device)
            predictions = model(e1,e2).squeeze(1)
            pred = np.array(predictions.data.tolist())
            real = np.array(batch.meanGrade.data.tolist())
            myid = np.array(batch.id.data.tolist())
            real_list = np.append(real_list, real)
            pred_list = np.append(pred_list, pred)
            id_list = np.append(id_list, myid)

    
# #     csv['pred_label'] = pred_list.round(0).astype(int)
#     csv['pred'] = [Result_Dict[i] for i in csv['pred_label']]


    df = pd.DataFrame({'id':id_list, 'real':real_list, 'pred':pred_list})
    rmse = np.sqrt(np.mean((df['real'] - df['pred'])**2))
            
    print(rmse)
    return df


# # 测试 best的model

# In[ ]:


class RMSEPlus():
    """
    for full: RMSE
    for top n% + bottom n%: RMSE@10, RMSE@20, RMSE@30, RMSE@40
    """

    def __init__(self):
        self.pred_list = []
        self.real_list = []

    def __call__(self, predictions, labels):
        if isinstance(predictions, torch.Tensor):
            #predictions = predictions.detach().cpu().numpy()
            predictions = predictions.data.tolist()
        if isinstance(labels, torch.Tensor):
            #labels = labels.detach().cpu().numpy()
            labels = labels.data.tolist()

        self.real_list += labels
        self.pred_list += predictions

    def get_metric(self, reset=False):
        metrics = {}
        df = pd.DataFrame({'real': self.real_list, 'pred': self.pred_list})
        metrics['rmse'] = np.sqrt(np.mean((df['real'] - df['pred']) ** 2))

        df = df.sort_values(by=['real'], ascending=False)
        for percent in [10, 20, 30, 40]:
            size = math.ceil(len(df) * percent * 0.01)
            # top n % + bottom n %
            df2 = df[:size].append(df[-size:])
            rmse = np.sqrt(np.mean((df2['real'] - df2['pred'])**2))
            metrics[f'rmse_{percent}'] = rmse
        if reset:
            self.reset()

        return metrics

    def reset(self):
        self.pred_list = []
        self.real_list = []


# In[27]:


model.load_state_dict(torch.load("lstm-model.pth"))
#df = myMSE(model, train_iterator)


# In[26]:


df = myMSE(model, valid_iterator)


# In[44]:


df = myMSE(model, test_iterator)


# In[41]:


import math
test = RMSEPlus()
m = test.get_metric()


# In[ ]:




