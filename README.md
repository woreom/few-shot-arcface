# recognize-this
image recognition model maker
# How To Use
### Instalation
```
pip install -r requirements.txt
```
or
```
conda env create -f environment.yml
```
### Prepare dateset
Preprocess images and put them in datasets folder
```
datasets
└───dataset_name
│   └─── class1
│   └─── class2
│   └─── class3
│       │   image1.jpg
│       │   image2.jp
│       │   ...
│   └───class4
|   ```
```
### Train model
1. open BackBoneTest.ipynb
2. choose backbone or write your own
3. make tfrecord of your dataset
4. train embeddings model with your backbone
5. make embeddings database with the embeddings model 
5. train classifier model with with the embeddings 

