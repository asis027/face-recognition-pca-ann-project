import os, cv2, pickle, argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pca_ann import FaceRecognizer

def load_dataset(root, size):
    X, y = [], []
    for person in os.listdir(root):
        pd = os.path.join(root, person)
        if not os.path.isdir(pd): continue
        for img in os.listdir(pd):
            im = cv2.imread(os.path.join(pd,img),0)
            if im is None: continue
            im = cv2.resize(im,(size,size)).flatten()
            X.append(im); y.append(person)
    return np.array(X), np.array(y)

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", required=True)
ap.add_argument("--classifier", default="mlp")
ap.add_argument("--output", default="model_output")
args = ap.parse_args()

X,y = load_dataset(args.data_dir,64)
Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)

model = FaceRecognizer(classifier=args.classifier)
model.fit(Xtr,ytr)

pred,_ = model.predict(Xte)
print(classification_report(yte,pred))

os.makedirs(args.output,exist_ok=True)
with open(os.path.join(args.output,"model.pkl"),"wb") as f:
    pickle.dump(model,f)

print("Model saved")