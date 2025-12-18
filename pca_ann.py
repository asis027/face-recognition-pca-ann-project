import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

class FaceRecognizer:
    def __init__(self, k=50, classifier="mlp"):
        self.encoder = LabelEncoder()
        self.pca = PCA(n_components=k, whiten=True, random_state=42)
        if classifier == "svm":
            self.clf = SVC(probability=True)
        elif classifier == "knn":
            self.clf = KNeighborsClassifier(n_neighbors=3)
        else:
            self.clf = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=500)

    def fit(self, X, y):
        y_enc = self.encoder.fit_transform(y)
        Xp = self.pca.fit_transform(X)
        self.clf.fit(Xp, y_enc)

    def predict(self, X):
        Xp = self.pca.transform(X)
        probs = self.clf.predict_proba(Xp)
        idx = probs.argmax(axis=1)
        return self.encoder.inverse_transform(idx), probs.max(axis=1)