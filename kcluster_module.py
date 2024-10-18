import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import nltk

class TextProcessingAndDatabase:
    def __init__(self, dataset_handler):
        """
        Initialize the text processor with a reference to DatasetHandler.
        """
        self.texts = []  # Store the processed texts
        self.dataset_handler = dataset_handler

    def lemmatize_text(self, text):
        lemmatizer = nltk.stem.WordNetLemmatizer()
        words = nltk.word_tokenize(text)
        lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalpha()]
        return ' '.join(lemmatized_words)

    def process_texts(self):
        """
        Load and process texts using the DatasetHandler.
        """
        try:
            # Load texts from the DatasetHandler (assuming 'Text' is the column name)
            df = self.dataset_handler.read_csv()
            self.texts = [self.lemmatize_text(text) for text in df['Text'].astype(str)]
            return df, self.texts
        except Exception as e:
            print(f"Error processing texts: {e}")
        return None, []

class KclusterAnalysis:
    def __init__(self, n_components=3, random_state=42):
        """
        Initializes KMeans and PCA models for K-cluster analysis.
        """
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=n_components, random_state=random_state)

    def calculate_wcss_and_silhouette(self, tfidf_matrix, max_clusters=200):
        """
        Calculate WCSS and silhouette scores for multiple clusters to find the optimal number of clusters.
        """
        wcss = []
        silhouette_scores = []
        for i in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=i, random_state=self.random_state)
            kmeans.fit(tfidf_matrix)
            wcss.append(kmeans.inertia_)
            score = silhouette_score(tfidf_matrix, kmeans.labels_)
            silhouette_scores.append(score)
        return wcss, silhouette_scores

    def perform_analysis(self, texts, max_clusters=150):
        """
        Perform TF-IDF, clustering, and PCA in one method.
        """
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.57, min_df=0.02, ngram_range=(1, 2))
        tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

        num_samples = tfidf_matrix.shape[0]
        max_clusters = min(max_clusters, num_samples)

        wcss, silhouette_scores = self.calculate_wcss_and_silhouette(tfidf_matrix, max_clusters)
        optimal_clusters = np.argmax(silhouette_scores) + 2

        kmeans = KMeans(n_clusters=min(optimal_clusters, max_clusters), random_state=self.random_state)
        labels = kmeans.fit_predict(tfidf_matrix)

        X_pca = self.pca.fit_transform(tfidf_matrix.toarray())

        return X_pca, labels

class KclusterCSVDataSaver:
    def save_analysis_to_csv(self, pca_results, labels, dataset_handler, output_filename="pca_output.csv"):
        """
        Convert PCA results and cluster labels to a CSV and save it.
        """
        df_pca = pd.DataFrame(pca_results, columns=[f'PCA{i+1}' for i in range(pca_results.shape[1])])
        df_pca['Kcluster_topic'] = labels  # Add the cluster labels as a new column
        df_pca['Kcluster_PCA'] = df_pca.apply(lambda row: ', '.join(map(str, row[:-1])), axis=1)  # Combine PCA components

        df = dataset_handler.read_csv()  # Load existing CSV
        df['Kcluster_PCA'] = df_pca['Kcluster_PCA']
        df['Kcluster_topic'] = df_pca['Kcluster_topic']

        dataset_handler.write_csv(df)
        print(f"PCA and Cluster results saved to the CSV.")