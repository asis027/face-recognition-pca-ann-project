# Face Recognition using PCA + ANN / SVM / KNN

## Features
- Eigenfaces (PCA)
- ANN, SVM, KNN classifiers
- Unknown face detection
- Evaluation & confusion matrix
- Eigenfaces visualization

## Run
python train.py --data_dir dataset --classifier mlp
python predict.py --model model_output/model.pkl --image test.jpg
python evaluate.py --model model_output/model.pkl --data_dir dataset