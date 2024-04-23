import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.stats import mode
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

MAX_LEN = 30
BATCH_SIZE = 32
EPOCHS = 10

class CustomDataset(Dataset):
    """
    Custom dataset class for text classification.

    Args:
        dataframe (DataFrame): The input dataframe containing 'text' and 'label' columns.
        tokenizer (BertTokenizer): The tokenizer to be used for encoding the text.
        max_len (int): The maximum length of the input text.

    Returns:
        dict: A dictionary containing the input_ids, attention_mask, and label tensors.

    """
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
    """
    Custom dataset class for testing without labels.

    Args:
        dataframe (DataFrame): The input dataframe containing 'text' column.
        tokenizer (BertTokenizer): The tokenizer to be used for encoding the text.
        max_len (int): The maximum length of the input text.

    Returns:
        dict: A dictionary containing the input_ids and attention_mask tensors.

    """
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

def BagOfWords(df_train, df_test):
    """
    Perform Bag of Words text classification using Multinomial Naive Bayes.

    Args:
        df_train (DataFrame): The training dataset containing 'text' and 'label' columns.
        df_test (DataFrame): The testing dataset containing 'text' and 'label' columns.

    Returns:
        MultinomialNB: The trained classifier.

    """
    # Extracting the features and labels from the training and testing datasets
    X_train = df_train['text']
    y_train = df_train['label']
    X_test = df_test['text']
    y_test = df_test['label']

    # Vectorization of the text
    vectorizer = CountVectorizer(stop_words='english')
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    # Multinomial Naive Bayes for classification
    classifier = MultinomialNB()
    classifier.fit(X_train_vectorized, y_train)

    # Making predictions on the testing dataset
    predictions = classifier.predict(X_test_vectorized)

    # Calculating the accuracy of the classifier
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy)
    return classifier

def BertFineTuning(df_train, df_test):
    """
    Perform fine-tuning of BERT model for sequence classification.

    Args:
        df_train (DataFrame): The training dataset containing 'text' and 'label' columns.
        df_test (DataFrame): The testing dataset containing 'text' and 'label' columns.

    Returns:
        BertForSequenceClassification: The trained BERT model.

    """
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
    """
    Generate predictions using the fine-tuned BERT model.

    Args:
        model (BertForSequenceClassification): The trained BERT model.
        df_test (DataFrame): The testing dataset containing 'text' column.

    Returns:
        list: The predicted labels.

    """
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
    """
    Create sentence embeddings using the MiniLM model.

    Args:
        df_train_aug (DataFrame): The augmented training dataset containing 'text' column.
        df_test (DataFrame): The testing dataset containing 'text' column.

    Returns:
        torch.Tensor: The embeddings of the training dataset.
        torch.Tensor: The embeddings of the testing dataset.

    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    train_embeddings = model.encode(df_train_aug['text'], show_progress_bar=True)
    test_embeddings = model.encode(df_test['text'], show_progress_bar=True)

    train_embeddings = torch.tensor(train_embeddings)
    test_embeddings = torch.tensor(test_embeddings)

    torch.save(train_embeddings, 'train_embeddings_100_entier.pt')
    torch.save(test_embeddings, 'test_embeddings_100_entier.pt')
    
    return train_embeddings, test_embeddings

def ensemble_learning(train_embeddings, test_embeddings, df_train_aug):
    """
    Perform ensemble learning using multiple classifiers.

    Args:
        train_embeddings (torch.Tensor): The embeddings of the training dataset.
        test_embeddings (torch.Tensor): The embeddings of the testing dataset.
        df_train_aug (DataFrame): The augmented training dataset containing 'label' column.

    Returns:
        list: The ensemble predictions.

    """
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
