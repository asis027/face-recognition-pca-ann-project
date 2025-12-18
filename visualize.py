import pickle,argparse,matplotlib.pyplot as plt

ap=argparse.ArgumentParser()
ap.add_argument("--model",required=True)
ap.add_argument("--n",type=int,default=16)
args=ap.parse_args()

with open(args.model,"rb") as f:
    model=pickle.load(f)

faces=model.pca.components_[:args.n]
fig,ax=plt.subplots(4,4)
for a,f in zip(ax.flatten(),faces):
    a.imshow(f.reshape(64,64),cmap="gray")
    a.axis("off")
plt.show()