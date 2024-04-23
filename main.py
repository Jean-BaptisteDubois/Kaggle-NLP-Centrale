from preprocessing import load_data, preprocess_data
import torch 
from models import creation_embeddings_llm, ensemble_learning, BertFineTuning, bert_predictions, BagOfWords
import torch
import pandas as pd


path_train = './data/train.json'
path_test = './data/test_shuffle.txt'


if __name__ == '__main__':
    df_train, df_test = load_data(path_train, path_test)
    #df_train, df_test = preprocess_data(df_train, df_test) #if stopping words and lemmatization are needed
    #df_train_aug = pd.read_csv('augmented_data.csv') #If data augmentation is desired
    
    """Bag of Words"""
    #The model is already trained
    model = BagOfWords(df_train, df_test)
    
    
    """Bert Fine-tuning"""
    model = BertFineTuning(df_train, df_test)
    predictions = bert_predictions(model, df_test)
    
    
    """Ensemble learning with the embeddings of the model MiniLM-L6-v2"""
    
    #train_embeddings, test_embeddings = creation_embeddings_llm(df_train, df_test)
    train_embeddings = torch.load('./data/train_embeddings_3_entier.pt')
    test_embeddings = torch.load('./data/test_embeddings_3_entier.pt')
    #The visualization can be seen in the noteebook
    #visualize_embeddings(train_embeddings, test_embeddings, df_train_aug, le)
    #visualize_embeddings_tsne(train_embeddings, le)
    ensemble_predictions = ensemble_learning(train_embeddings, test_embeddings, df_train)
    
    
    
    