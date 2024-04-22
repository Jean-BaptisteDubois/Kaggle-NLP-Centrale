import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nlpaug.augmenter.word as naw
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json

def load_data(file_path_train, file_path_test):
    """
    Load data from a file and return train and test dataframes.

    Args:
        file_path_train (str): The file path of the training data.
        file_path_test (str): The file path of the test data.
    Returns:
        df_train (pandas.DataFrame): The dataframe containing the training data.
        df_test (pandas.DataFrame): The dataframe containing the test data.
    """
    with open(file_path_train, 'r') as f:
        data_train = json.load(f)
    
    with open(file_path_test, 'r', encoding='utf-8') as f:
        data_test = f.readlines()
    
    rows = []
    for label, texts in data_train.items():
        for text in texts:
            rows.append({'label': label, 'text': text})
    df_train = pd.DataFrame(rows)
    
    rows = []
    for data in data_test:
        rows.append({'text': data})
    
    df_test = pd.DataFrame(rows)
    le = LabelEncoder()
    df_train['label'] = le.fit_transform(df_train['label'])
    df_test['label'] = le.transform(df_test['label'])
    
    return df_train, df_test

def clean_text(text):
    """
    Clean the given text by removing stopwords, lemmatizing words, and converting to lowercase.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    text = text.lower()
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word.isalnum()]
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

def preprocess_data(df_train, df_test):
    """
    Preprocess the given dataframes by cleaning the text data.

    Args:
        df_train (pandas.DataFrame): The dataframe containing the training data.
        df_test (pandas.DataFrame): The dataframe containing the test data.

    Returns:
        df_train (pandas.DataFrame): The preprocessed dataframe containing the training data.
        df_test (pandas.DataFrame): The preprocessed dataframe containing the test data.
    """
    df_train['text'] = df_train['text'].apply(clean_text)
    df_test['text'] = df_test['text'].apply(clean_text)
    return df_train, df_test



def data_augmentation(df_train):
    """
    Perform data augmentation on the given dataframe by generating augmented texts.

    Args:
        df_train (pandas.DataFrame): The dataframe containing the training data.

    Returns:
        pandas.DataFrame: The dataframe containing the augmented training data.
    """
    aug1 = naw.SynonymAug(aug_src='wordnet')
    #aug2 = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert")
    #aug2 = naw.BackTranslationAug()
    #aug = naw.ContextualWordEmbsAug(model_path=, action="substitute")
    #aug = nas.ContextualWordEmbsForSentenceAug(model_path='gpt2')

    augmented_texts = []
    num_augmentations = 100
    for text, label in zip(df_train['text'], df_train['label']):
        for _ in range(num_augmentations):
            augmented_text = aug1.augment(text)
            augmented_texts.append({'text': augmented_text, 'label': label})
    augmented_train_df = pd.DataFrame(augmented_texts)
    final_train_df = pd.concat([df_train, augmented_train_df], ignore_index=True)
    final_train_df.to_csv('train_augmented_100.csv', index=False)
    return final_train_df