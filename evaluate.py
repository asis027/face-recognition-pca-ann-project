import os,cv2,pickle,argparse
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns

ap=argparse.ArgumentParser()
ap.add_argument("--model",required=True)
ap.add_argument("--data_dir",required=True)
args=ap.parse_args()

with open(args.model,"rb") as f:
    model=pickle.load(f)

X,y=[],[]
for p in os.listdir(args.data_dir):
    for img in os.listdir(os.path.join(args.data_dir,p)):
        im=cv2.imread(os.path.join(args.data_dir,p,img),0)
        if im is None: continue
        im=cv2.resize(im,(64,64)).flatten()
        X.append(im); y.append(p)

pred,_=model.predict(np.array(X))
print(classification_report(y,pred))

cm=confusion_matrix(y,pred,labels=model.encoder.classes_)
sns.heatmap(cm,annot=True,fmt="d",xticklabels=model.encoder.classes_,
            yticklabels=model.encoder.classes_)
plt.show()