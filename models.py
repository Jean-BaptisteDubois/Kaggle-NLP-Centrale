import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.stats import mode
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW



MAX_LEN = 30
BATCH_SIZE = 32
EPOCHS = 10

class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        label = self.data.iloc[idx]['label']
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
        
class CustomDatasetTest(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }


def BertFineTuning(df_train, df_test):
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = CustomDataset(df_train, tokenizer, MAX_LEN)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=12)

    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    optimizer = AdamW(model.parameters(), lr=1e-5)
    losses = []
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            attention_mask = batch['attention_mask'].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            labels = batch['label'].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_train_loss = total_loss / len(train_dataloader)
        losses.append(avg_train_loss)
        print(f'Epoch {epoch+1}/{EPOCHS}')
        print(f'Training loss: {avg_train_loss:.4f}')
    torch.save(model.state_dict(), 'model_bert.pt')
    return model

def bert_predictions(model, df_test):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    test_dataset = CustomDatasetTest(df_test, tokenizer, MAX_LEN)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    predictions = []
    for batch in test_dataloader:
        input_ids = batch['input_ids'].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        attention_mask = batch['attention_mask'].to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions.extend(torch.argmax(logits, dim=1).tolist())
    return predictions



def creation_embeddings_llm(df_train_aug, df_test):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    train_embeddings = model.encode(df_train_aug['text'], show_progress_bar=True)
    test_embeddings = model.encode(df_test['text'], show_progress_bar=True)

    train_embeddings = torch.tensor(train_embeddings)
    test_embeddings = torch.tensor(test_embeddings)

    torch.save(train_embeddings, 'train_embeddings_100_entier.pt')
    torch.save(test_embeddings, 'test_embeddings_100_entier.pt')
    
    return train_embeddings, test_embeddings


def ensemble_learning(train_embeddings, test_embeddings, df_train_aug):
    weights = {i: 1 for i in range(12)}
    weights[5] = 2
    list_classifiers = [SGDClassifier(loss='hinge', penalty='l2',alpha=0.005, random_state=42,class_weight=weights),
                        LogisticRegression(max_iter=1000, random_state=42,class_weight=weights),
                        RandomForestClassifier(random_state=42, class_weight=weights),
                        SVC(probability=True, random_state=42, class_weight=weights),]

    for classifier in list_classifiers:
        classifier.fit(train_embeddings, df_train_aug['label'])
        print(f'{classifier.__class__.__name__} trained')


    list_predictions = []
    for classifier in list_classifiers:
        predictions = classifier.predict(test_embeddings)
        list_predictions.append(predictions)
    ensemble_predictions = mode(list_predictions)[0][0]
    return ensemble_predictions