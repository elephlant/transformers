import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import transformers as ppb
import warnings
warnings.filterwarnings('ignore')

from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

# from tqdm import tqdm
from make_logger import make_logger

context = 'DistilBERT-SST2-transfer'
logger = make_logger('HuggingFace', 'logs/', context)
logger.info('Running transfer learning experiments on SST2 dataset, with various versions of BERT.')

class SST2Dataset(Dataset):
    """SST2 dataset. Movie reviews."""

    def __init__(self, tsv_file, tokenizer):
        """
        Args:
            tsv_file (string): Path to the tsv file with annotations.
            tokenizer (tokenizer): The tokenizer.
        """
        # data = pd.read_csv(tsv_file, delimiter='\t', header=None)[:2000] # TODO
        data = pd.read_csv(tsv_file, delimiter='\t', header=None)
        tokenized = data[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

        max_len = 0
        for i in tokenized.values:
            if len(i) > max_len:
                max_len = len(i)

        self.features = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
        self.attention_mask = np.where(self.features != 0, 1, 0)
        self.labels = np.array(data[1])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        features = torch.tensor(self.features[idx])
        labels = torch.tensor(self.labels[idx])
        masks = torch.tensor(self.attention_mask[idx])

        features = Variable(features)
        labels = Variable(labels)
        masks = Variable(masks)

        return {'features': features, 'labels': labels, 'masks': masks}

train_tsv = 'data/SST2/train.tsv'
dev_tsv = 'data/SST2/dev.tsv'
test_tsv = 'data/SST2/test.tsv'
# df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv', delimiter='\t', header=None)
# df = pd.read_csv(, delimiter='\t', header=None)

# df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/dev.tsv', delimiter='\t', header=None)

# print('Writing to own folder...')
# df.to_csv('data/SST2/dev.tsv', sep='\t', header=None, index=False)
# print('Done')
# exit()

# batch_1 = df[:2000]
# print()
# print(batch_1[1].value_counts())
# print()

print('Loading pretrained model...')
# For DistilBERT:
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

## Want BERT instead of distilBERT? Uncomment the following line:
# model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)
# device = 'cpu' #TODO
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)
print('Done. Loading dataset...')

train_dataset = SST2Dataset(train_tsv, tokenizer)
dev_dataset = SST2Dataset(dev_tsv, tokenizer)
test_dataset = SST2Dataset(test_tsv, tokenizer)

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=8)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size,shuffle=False, num_workers=8)
test_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=False, num_workers=8)

print('Done.')

# for i, sample_batched in enumerate(dataloader):
#     features, labels, masks = sample_batched['features'], sample_batched['labels'], sample_batched['masks']
#     print(type(features))
#     print(features.size())
#     print(labels.size())
#     print(masks.size())
#     exit()

# sample = dataset[0]
# print(sample)
# exit()
# tokenized = batch_1[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

# max_len = 0
# for i in tokenized.values:
#     if len(i) > max_len:
#         max_len = len(i)

# padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
# attention_mask = np.where(padded != 0, 1, 0)

# input_ids = torch.tensor(padded)  
# attention_mask = torch.tensor(attention_mask)
# labels = torch.tensor(batch_1[1])

# if torch.cuda_is_available():
#     input_ids = Variable(input_ids.cuda())
#     labels = Variable(labels.cuda())
#     attention_mask = Variable(attention_mask.cuda())
# else:
#     input_ids = Variable(input_ids)
#     labels = Variable(labels)
#     attention_mask = Variable(attention_mask)

# print('forward()ing...')
# with torch.no_grad():
#     last_hidden_states = model(input_ids, attention_mask=attention_mask)

# features = last_hidden_states[0][:,0,:].numpy()
# labels = batch_1[1]

# train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

# parameters = {'C': np.linspace(0.0001, 100, 20)}
# grid_search = GridSearchCV(LogisticRegression(max_iter=1000000), parameters)
# grid_search.fit(train_features, train_labels)

# print('best parameters: ', grid_search.best_params_)
# print('best scrores: ', grid_search.best_score_)

# lr_clf = LogisticRegression(max_iter=1000000, C=grid_search.best_params_['C'])
# lr_clf.fit(train_features, train_labels)

# print('score:',lr_clf.score(test_features, test_labels))

class BERTTransferClassifier(nn.Module):
    def __init__(self, bert_model, n_classes, d_model, d_hidden=2048):
        super(BERTTransferClassifier, self).__init__()
        self.bert_model = bert_model
        self.ffn1 = nn.Linear( d_model, d_hidden )
        self.relu = nn.ReLU()
        self.ffn2 = nn.Linear( d_hidden, n_classes )
        self.bert_training = False
    
    def train_bert(self):
        self.bert_training = True
    
    def freeze_bert(self):
        self.bert_training = False
    
    # def params(self):
    #     params = [self.ffn1.parameters(), self.ffn2.parameters()]
    #     if self.bert_training:
    #         params.append(self.bert_model.parameters())
    #     return params
    
    def forward(self, x, attention_mask):
        if self.bert_training:
            x = self.bert_model(x, attention_mask)
        else:
            with torch.no_grad():
                x = self.bert_model(x, attention_mask)
        # extract the features from the CLS part only
        x = x[0][:,0,:]

        x = self.ffn1(x)
        x = self.relu(x)
        logits = self.ffn2(x)
        return logits

print('Creating classifier...')
clf = BERTTransferClassifier( model, n_classes=2, d_model=768  )
clf = clf.to(device)
print('Constructing criterion and optimizer')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(clf.parameters(), lr=0.001)

n_warmup_epochs = 7
n_epochs = 20

report_freq = 5

clf.freeze_bert()
for epoch in range(n_warmup_epochs):
    running_loss = 0.
    for i, sample_batched in enumerate(train_dataloader):
        features, labels, masks = sample_batched['features'], sample_batched['labels'], sample_batched['masks']
        
        # if torch.cuda.is_available():
        #     features = features.cuda()
        #     labels = labels.cuda()
        #     masks = masks.cuda()
        features = features.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        out = clf(features, masks)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % report_freq == (report_freq-1):
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / report_freq))
            running_loss = 0.0
    
    total_val_loss = 0.
    total_val_steps = 0
    print('Computing val loss...')
    for i, sample_batched in enumerate(dev_dataloader):
        features, labels, masks = sample_batched['features'], sample_batched['labels'], sample_batched['masks']
        features = features.to(device)
        labels = labels.to(device)
        masks = masks.to(device)

        total_val_steps += 1
        with torch.no_grad():
            out = clf(features, masks)
            loss = criterion(out, labels)
            total_val_loss += loss
    print('val loss: %.3f' % (total_val_loss.item()/total_val_steps) )

print('Warmed up. Now training BERT as well...')
clf.train_bert()
del optimizer
optimizer = torch.optim.Adam(clf.parameters(), lr=0.00001)
for epoch in range(n_epochs):
    running_loss = 0.
    for i, sample_batched in enumerate(train_dataloader):
        features, labels, masks = sample_batched['features'], sample_batched['labels'], sample_batched['masks']
        features = features.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        out = clf(features, masks)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % report_freq == (report_freq-1):
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / report_freq))
            running_loss = 0.0
    
    total_val_loss = 0.
    total_val_steps = 0
    print('Computing val loss...')
    for i, sample_batched in enumerate(dev_dataloader):
        features, labels, masks = sample_batched['features'], sample_batched['labels'], sample_batched['masks']
        features = features.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        total_val_steps += 1
        with torch.no_grad():
            out = clf(features, masks)
            loss = criterion(out, labels)
            total_val_loss += loss
    print('val loss: %.3f' % (total_val_loss.item()/total_val_steps) )

print('Finished training')

# Testing and reporting accuracy of trained model
print('Computing test accuracy...')
total_correct = 0
total_test = 0
for i, sample_batched in enumerate(test_dataloader):
    features, labels, masks = sample_batched['features'], sample_batched['labels'], sample_batched['masks']
    features = features.to(device)
    labels = labels.to(device)
    masks = masks.to(device)
    total_test += len(labels)
    with torch.no_grad():
        out = clf(features, masks)
        probs = nn.Softmax(dim=-1)(out)
        preds = torch.argmax(probs,dim=-1)
        # print(preds)
        # print(labels)
        total_correct += torch.sum((preds == labels).float())

lines = []
print('total correct:', total_correct.item())
lines.append( 'total correct: {}'.format(total_correct.item()) )
print('total in test:', total_test)
lines.append( 'total in test: {}'.format(total_test)  )
print('test acc:', float(total_correct) / total_test)
lines.append( 'test acc: {}'.format(float(total_correct) / total_test) )

with open('log.txt', 'w') as f:
    f.write('\n'.join(lines))