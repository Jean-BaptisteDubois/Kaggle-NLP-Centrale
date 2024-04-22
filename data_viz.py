from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_embeddings(train_embeddings, test_embeddings, df_train_aug, le):
    pca = PCA(n_components=3)
    train_embeddings_pca = pca.fit_transform(train_embeddings)
    test_embeddings_pca = pca.transform(test_embeddings)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for label in df_train_aug['label'].unique():
        indices = df_train_aug['label'] == label
        ax.scatter(train_embeddings_pca[indices, 0], train_embeddings_pca[indices, 1], train_embeddings_pca[indices, 2], label=le.inverse_transform([label])[0])
    ax.legend()
    plt.show()
    

def visualize_embeddings_tsne(train_embeddings, le):
    tsne = TSNE(n_components=2, random_state=0)
    train_proj = tsne.fit_transform(train_embeddings)

    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=train_proj[:, 0], y=train_proj[:, 1], hue=le.df_train_aug['label'], palette='tab10')
    plt.title('t-SNE projection of the BERT embeddings')
    plt.show()