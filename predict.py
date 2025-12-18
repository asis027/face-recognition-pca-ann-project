import cv2,pickle,argparse
import numpy as np

ap=argparse.ArgumentParser()
ap.add_argument("--model",required=True)
ap.add_argument("--image",required=True)
ap.add_argument("--threshold",type=float,default=0.5)
args=ap.parse_args()

with open(args.model,"rb") as f:
    model=pickle.load(f)

img=cv2.imread(args.image,0)
img=cv2.resize(img,(64,64)).flatten().reshape(1,-1)

label,conf=model.predict(img)
if conf[0]<args.threshold:
    print("Unknown person")
else:
    print("Prediction:",label[0],"Confidence:",round(conf[0],2))