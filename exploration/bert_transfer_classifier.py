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

from make_logger import make_logger

# Dataset class for SST2 (https://nlp.stanford.edu/sentiment/)
class SST2Dataset(Dataset):
    """SST2 dataset. Movie reviews."""

    def __init__(self, tsv_file, tokenizer):
        """
        Args:
            tsv_file (string): Path to the tsv file with annotations.
            tokenizer (tokenizer): The tokenizer.
        """
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

class BERTTransferClassifier(nn.Module):
    def __init__(self, bert_model, n_classes, d_model, d_hidden=2048):
        super(BERTTransferClassifier, self).__init__()
        self.basemodel = bert_model
        self.dropout1 = nn.Dropout(0.1)
        self.ffn1 = nn.Linear( d_model, d_hidden )
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.ffn2 = nn.Linear( d_hidden, n_classes )
        self.bert_training = False
    
    def train_base(self):
        self.bert_training = True
    
    def freeze_base(self):
        self.bert_training = False
        
    def forward(self, x, attention_mask):
        if self.bert_training:
            x = self.basemodel(x, attention_mask)[0]
        else:
            with torch.no_grad():
                x = self.basemodel(x, attention_mask)[0]
        
        x = self.dropout1(x)
        
        # extract the features from the CLS part only
        x = x[:,0,:]

        x = self.ffn1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        logits = self.ffn2(x)
        return logits

def train(model, n_epochs, train_dataloader, dev_dataloader, device, optimizer, criterion, logger, report_freq=50):
    for epoch in range(n_epochs):
        running_loss = 0.
        for i, sample_batched in enumerate(train_dataloader):
            features, labels, masks = sample_batched['features'], sample_batched['labels'], sample_batched['masks']
            
            features = features.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            out = model(features, masks)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % report_freq == (report_freq-1):
                logger.info('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / report_freq))
                running_loss = 0.0
        
        total_val_loss = 0.
        total_val_steps = 0
        logger.info('Computing val loss...')
        for i, sample_batched in enumerate(dev_dataloader):
            features, labels, masks = sample_batched['features'], sample_batched['labels'], sample_batched['masks']
            features = features.to(device)
            labels = labels.to(device)
            masks = masks.to(device)

            total_val_steps += 1
            with torch.no_grad():
                out = model(features, masks)
                loss = criterion(out, labels)
                total_val_loss += loss
        logger.info('Val loss: %.3f' % (total_val_loss.item()/total_val_steps) )

if __name__ == '__main__':
    context = 'BERT-SST2-transfer-dropout'
    logger = make_logger('HuggingFace', 'logs/', context)
    logger.info('Running transfer learning experiments on SST2 dataset, currently on context: '.format(context))

    train_tsv = 'data/SST2/train.tsv'
    dev_tsv = 'data/SST2/dev.tsv'
    test_tsv = 'data/SST2/test.tsv'

    # Where the data is from
    # df = pd.read_csv('https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv', delimiter='\t', header=None)
    # df = pd.read_csv(, delimiter='\t', header=None)

    # print('Writing to own folder...')
    # df.to_csv('data/SST2/dev.tsv', sep='\t', header=None, index=False)
    # print('Done')
    # exit()

    # batch_1 = df[:2000]
    # print()
    # print(batch_1[1].value_counts())
    # print()

    logger.info('Loading pretrained model...')

    # For distilBERT:
    # model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
    model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
    '''
    ## Want BERT instead of distilBERT? Uncomment the following line:
    # model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
    # Transformers has a unified API
    # for 10 transformer architectures and 30 pretrained weights.
    #          Model          | Tokenizer          | Pretrained weights shortcut
    MODELS = [(BertModel,       BertTokenizer,       'bert-base-uncased'),
            (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),
            (GPT2Model,       GPT2Tokenizer,       'gpt2'),
            (CTRLModel,       CTRLTokenizer,       'ctrl'),
            (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103'),
            (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
            (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
            (DistilBertModel, DistilBertTokenizer, 'distilbert-base-cased'),
            (RobertaModel,    RobertaTokenizer,    'roberta-base'),
            (XLMRobertaModel, XLMRobertaTokenizer, 'xlm-roberta-base'),
            ]
    '''

    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    basemodel = model_class.from_pretrained(pretrained_weights)
    logger.info('Making train/val/test datasets and dataloaders...')

    train_dataset = SST2Dataset(train_tsv, tokenizer)
    dev_dataset = SST2Dataset(dev_tsv, tokenizer)
    test_dataset = SST2Dataset(test_tsv, tokenizer)

    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True, num_workers=8)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size,shuffle=False, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False, num_workers=8)

    # To test that the dataloaders work.
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # logger.info('Creating classifier...')
    model = BERTTransferClassifier( basemodel, n_classes=2, d_model=768  )
    model = model.to(device)
    # logger.info('Constructing criterion and optimizer')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_warmup_epochs = 3
    n_epochs = 2

    model.freeze_base()
    logger.info('Warming up model by training classifier layer first...')

    train(model, n_warmup_epochs, train_dataloader, dev_dataloader, device, optimizer, criterion, logger)

    logger.info('Warmed up. Now training the basemodel jointly...')
    # Let backprop train the base model inside
    model.train_base()

    # Use a fresh optimizer with a much smaller learning rate
    del optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    train(model, n_epochs, train_dataloader, dev_dataloader, device, optimizer, criterion, logger)

    logger.info('Completed training.')

    # Testing and reporting accuracy of trained model
    logger.info('Computing test accuracy...')
    total_correct = 0
    total_test = 0
    for i, sample_batched in enumerate(test_dataloader):
        features, labels, masks = sample_batched['features'], sample_batched['labels'], sample_batched['masks']
        features = features.to(device)
        labels = labels.to(device)
        masks = masks.to(device)
        total_test += len(labels)
        with torch.no_grad():
            out = model(features, masks)
            probs = nn.Softmax(dim=-1)(out)
            preds = torch.argmax(probs,dim=-1)
            total_correct += torch.sum((preds == labels).float())


    total_correct = total_correct.item()
    logger.info('Total correct: {}'.format(total_correct))
    logger.info('Total in test: {}'.format(total_test))
    logger.info('Test accuracy: {}'.format(float(total_correct) / total_test))
