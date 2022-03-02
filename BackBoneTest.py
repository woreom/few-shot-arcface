#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import pickle
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, train_test_split


from tensorflow import keras
import tensorflow as tf

from arcface.dataset import build_dataset
from arcface.losses import ArcLoss
from arcface.network import ArcLayer, L2Normalization, hrnet_v2, resnet101
from arcface.training_supervisor import TrainingSupervisor

from tensorflow.keras.applications.efficientnet import EfficientNetB3

import IPython.display as display
from glob import glob
from tqdm import tqdm
from abc import ABC, abstractmethod


# # TODO
# 
# 
# 1. hrnet (Done)
# 2. resnet ()
# 3. efficienthrnet
# 4. efficientnet
# 

# ### TFRecord

# In[2]:


def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()])
    )

def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def create_example(image, label):
    feature = {
        "image": image_feature(image),
        "label": bytes_feature(label),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

### Backbone Class make (ToDo Make into a separate module)############################
class RecordOperator(ABC):

    def __init__(self, filename):
        # Construct a reader if the user is trying to read the record file.
        self.dataset = None
        self._writer = None
        if tf.io.gfile.exists(filename):
            self.dataset = tf.data.TFRecordDataset(filename)
        else:
            # Construct a writer in case the user want to write something.
            self._writer = tf.io.TFRecordWriter(filename)

        # Set the feature description. This should be provided before trying to
        # parse the record file.
        self.set_feature_description()

    @abstractmethod
    def make_example(self):
        """Returns a tf.train.example from values to be saved."""
        pass

    def write_example(self, tf_example):
        """Create TFRecord example from a data sample."""
        if self._writer is None:
            raise IOError("Record file already exists.")
        else:
            self._writer.write(tf_example.SerializeToString())

    @abstractmethod
    def set_feature_description(self):
        """Set the feature_description to parse TFRecord file."""
        pass

    def parse_dataset(self):
        # Create a dictionary describing the features. This dict should be
        # consistent with the one used while generating the record file.
        if self.dataset is None:
            raise IOError("Dataset file not found.")

        def _parse_function(example_proto):
            # Parse the input tf.Example proto using the dictionary above.
            return tf.io.parse_single_example(example_proto, self.feature_description)

        parsed_dataset = self.dataset.map(_parse_function)
        return parsed_dataset
#########################################################################################

class ImageDataset(RecordOperator):
    """Construct ImageDataset tfrecord files."""

    def make_example(self, image, label):
        """Construct an tf.Example with image data and label.
        Args:
            image_string: encoded image, NOT as numpy array.
            label: the label.
        Returns:
            a tf.Example.
        """
        
        image_string = tf.image.decode_image(image)
        image_shape = image_string.shape
        

        # After getting all the features, time to generate a TensorFlow example.
        feature = {
            'image/height': int64_feature(image_shape[0]),
            'image/width': int64_feature(image_shape[1]),
            'image/depth': int64_feature(image_shape[2]),
            'image/encoded': image_feature(image_string),
            'label': int64_feature(label),
        }

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=feature))

        return tf_example
    
    def set_feature_description(self):
        self.feature_description = {
            "image/height": tf.io.FixedLenFeature([], tf.int64),
            "image/width": tf.io.FixedLenFeature([], tf.int64),
            "image/depth": tf.io.FixedLenFeature([], tf.int64),
            "image/encoded": tf.io.VarLenFeature(tf.float32),
            "label": tf.io.FixedLenFeature([], tf.int64),
        }
#         example = tf.io.parse_single_example(example, feature_description)
#         example["image"] = tf.io.decode_jpeg(example["image"], channels=3)
#         return example


def create_tfrecord(path = 'datasets/sorted_palmvein_roi/',  types=[".bmp"], tf_record='datasets/train.record'):
    
    converter = ImageDataset(tf_record)
    samples =[]
    [samples.extend(glob(path+"*/*."+typ)) for typ in types]
    total_samples_num = len(samples)
    ids = set()
    print("Total records: {}".format(total_samples_num))
    
    for i, image_path in tqdm(enumerate(samples)):
        image = tf.io.read_file(image_path)
        ids.add(image_path.split('/')[-2])
        label = len(ids)
        tf_example = converter.make_example(image, label)
        # Write the example to file.
        converter.write_example(tf_example)
        
    print("All done. Record file is:\n{}".format(tf_record))


# In[10]:


name='merge'
create_tfrecord(path= 'datasets/processed_merge/', types=['png','jpg','jepg','png', 'bmp'], tf_record='datasets/'+name+'_processed.record')


# # Training

# In[3]:


def train(base_model, name = "hrnetv2", train_files = "datasets/train.record", test_files = None, val_files = None, input_shape = (128, 128, 3),
          num_ids = 600, num_examples = 12000, training_dir = os.getcwd(),
          frequency = 1000, softmax = False, adam_alpha=0.001, adam_epsilon=0.001, batch_size = 8, export_only = False,
          override = False, epochs = 50, restore_weights_only=False):

    '''
    # Deep neural network training is complicated. The first thing is making
    # sure you have everything ready for training, like datasets, checkpoints,
    # logs, etc. Modify these paths to suit your needs.

    name:str = # What is the model's name?
    
    train_files:str = # Where are the training files?

    test_files:str = # Where are the testing files?

    val_files:str = # Where are the validation files? Set `None` if no files available.

    input_shape:tuple(int) = # What is the shape of the input image?

    embedding_size:int = # What is the size of the embeddings that represent the faces?

    num_ids:int = # How many identities do you have in the training dataset?

    num_examples:int = # How many examples do you have in the training dataset?

    # That should be sufficient for training. However if you want more
    # customization, please keep going.

    training_dir:str = # Where is the training direcotory for checkpoints and logs?

    regularizer = # Any weight regularization?

    frequency:int = # How often do you want to log and save the model, in steps?

    # All sets. Now it's time to build the model. There are two steps in ArcFace
    # training: 1, training with softmax loss; 2, training with arcloss. This
    # means not only different loss functions but also fragmented models.

    base_model:model = # First model is base model which outputs the face embeddings.
    '''
    
    # Where is the exported model going to be saved?
    export_dir = os.path.join(training_dir, 'exported', name)
    
    # Then build the second model for training.
    if softmax:
        print("Building training model with softmax loss...")
        model = keras.Sequential([keras.Input(input_shape),
                                  base_model,
                                  keras.layers.Dense(num_ids,
                                                     kernel_regularizer=regularizer),
                                  keras.layers.Softmax()],
                                 name="training_model")
        loss_fun = keras.losses.CategoricalCrossentropy()
    else:
        print("Building training model with ARC loss...")
        model = keras.Sequential([keras.Input(input_shape),
                                  base_model,
                                  L2Normalization(),
                                  ArcLayer(num_ids, regularizer)],
                                 name="training_model")
        loss_fun = ArcLoss()

    # Summary the model to find any thing suspicious at early stage.
    model.summary()

    # Construct an optimizer. This optimizer is different from the official
    # implementation which use SGD with momentum.
    optimizer = keras.optimizers.Adam(adam_alpha, amsgrad=True, epsilon=adam_epsilon)

    # Construct training datasets.
    dataset_train, dataset_val, test_dataset  = build_dataset(train_files,
                                  batch_size=batch_size,
                                  one_hot_depth=num_ids,
                                  training=True,
                                  val_size = 0.0,
                                  test_size = 0.0,
                                  num_examples = 12000,
                                  buffer_size=4096)

    # Construct dataset for validation. The loss value from this dataset can be
    # used to decide which checkpoint should be preserved.
    if val_files:
        dataset_val = build_dataset(val_files,
                                    batch_size=batch_size,
                                    one_hot_depth=num_ids,
                                    training=False,
                                    buffer_size=4096)

    # The training adventure is long and full of traps. A training supervisor
    # can help us to ease the pain.
    supervisor = TrainingSupervisor(model,
                                    optimizer,
                                    loss_fun,
                                    dataset_train,
                                    training_dir,
                                    frequency,
                                    "categorical_accuracy",
                                    'max',
                                    name)

    # If training accomplished, save the base model for inference.
    if export_only:
        print("The best model will be exported.")
        supervisor.restore(restore_weights_only, True)
        supervisor.export(base_model, export_dir)
        quit()

    # Restore the latest model if checkpoints are available.
    supervisor.restore(restore_weights_only)

    # Sometimes the training process might go wrong and we would like to resume
    # training from manually selected checkpoint. In this case some training
    # objects should be overridden before training started.
    if override:
        supervisor.override(0, 1)
        print("Training process overridden by user.")

    # Now it is safe to start training.
    supervisor.train(epochs, num_examples // batch_size)

    # Export the model after training.
    supervisor.export(base_model, export_dir)


# In[5]:


# First model is bexported/e model which outputs the face embeddings.
input_shape = (128, 128, 3)
embedding_size = 512
regularizer = keras.regularizers.L2(5e-4)
base_model = resnet101(input_shape=input_shape, output_size=embedding_size,
                           trainable=True, training=True,
                           kernel_regularizer=regularizer,
                           name="embedding_model")

name='merge'
train(base_model, name = name+"_soft", train_files = "datasets/"+name+"_processed.record", test_files = None, val_files = None, input_shape = (128, 128, 3),
          num_ids = 1144, num_examples = 30942, training_dir = 'outputs/',
          frequency = 1000, softmax = True, adam_alpha=0.001, adam_epsilon=0.001, batch_size = 8, export_only = False,
          override = False, epochs = 12, restore_weights_only=True)


# # Load softmax model and train with softmax = False

# In[9]:


checkpoint_dir = 'outputs/exported/processed_soft/'
num_ids = 600
regularizer = keras.regularizers.L2(5e-4)

base_model = keras.models.load_model(checkpoint_dir)


name='merge'
train(base_model, name = name+"_arc", train_files = "datasets/"+name+"_processed.record", test_files = None, val_files = None, input_shape = (128, 128, 3),
          num_ids = 1144, num_examples = 30942, training_dir = 'outputs/',
          frequency = 1000, softmax = False, adam_alpha=0.001, adam_epsilon=0.001, batch_size = 8, export_only = False,
          override = False, epochs = 12, restore_weights_only=True)


# # Classifier

# In[24]:


def save_pkl(pkl, path = 'model.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(pkl, f)
    print("saved pkl file at:",path)

def load_pkl(path='model.pkl'):
    with open(path, 'rb') as f:
        pkl = pickle.load(f)
    return pkl

def findCosineDistance(vector1, vector2):
    """
    Calculate cosine distance between two vector
    """
    vec1 = vector1.flatten()
    vec2 = vector2.flatten()

    a = np.dot(vec1.T, vec2)
    b = np.dot(vec1.T, vec1)
    c = np.dot(vec2.T, vec2)
    return 1 - (a/(np.sqrt(b)*np.sqrt(c)))

def cosine_similarity(test_vec, source_vecs):
    """
    Verify the similarity of one vector to group vectors of one class
    """
    cos_dist = 0
    for source_vec in source_vecs:
        cos_dist += findCosineDistance(test_vec, source_vec)
    return cos_dist/len(source_vecs)

def make_embeddings(dataset_path='datasets/sorted_palmvein_roi/', output_path='outputs/', model_dir='outputs/exported/arcface', types='bmp'):
    # Grab the paths to the input images in our dataset
    print("[INFO] quantifying palms...")
    imagePaths =[]
    [imagePaths.extend(glob(dataset_path+"*/*."+typ)) for typ in types]
#     imagePaths.extend(glob(dataset_path+'/*/*.jpg'))

    # Initialize model
    embedding_model = keras.models.load_model(model_dir)
       
    # Initialize our lists of extracted facial embeddings and corresponding people names
    knownEmbeddings = []
    knownNames = []

    # Initialize the total number of faces processed
    total = 0

    # Loop over the imagePaths
    for (i, imagePath) in tqdm(enumerate(imagePaths)):
        # extract the person name from the image path
        name = imagePath.split(os.path.sep)[-2]
#         print(imagePath)

         # load the image
        img = cv2.imread(imagePath).reshape(-1,128,128,3)
        palms_embedding = embedding_model.predict(img)[0]
        # add the name of the person + corresponding face
        # embedding to their respective list
        knownNames.append(name)
        knownEmbeddings.append(palms_embedding)
        total += 1
        
    print(total, " palms embedded")
#     print(set(knownNames))

    # save to output
    data = {"embeddings": knownEmbeddings, "names": knownNames}
    save_pkl(pkl=data, path=output_path+'db.pkl')
    
def make_model(embeddings_path='outputs/db.pkl', output_path='outputs/'):
    # Load the face embeddings
    data = load_pkl(embeddings_path)
    num_classes = len(np.unique(data["names"]))
    y = np.array(data["names"])
    X = np.array(data["embeddings"])
    
    
    # Initialize Softmax training model arguments
    input_shape = X.shape[1]
    cv = KFold(n_splits=5, random_state=1, shuffle=True)

    model =  MLPClassifier(hidden_layer_sizes=(input_shape, 640, 112, 640, num_classes), activation='tanh',max_iter=10000, batch_size='auto', learning_rate='adaptive',
                           validation_fraction=0.0, solver='adam', early_stopping=False ,verbose=0,random_state=1)

    for train_idx, valid_idx in cv.split(X):
        model.fit(X[train_idx], y[train_idx],)
        print(model.score(X[valid_idx], y[valid_idx]), end='\t')
    
    save_pkl(pkl=model, path=output_path+'model.pkl')
    
    return model


# In[25]:


make_embeddings(dataset_path='datasets/processed_merge/', output_path='outputs/exported/'+name+'_arc/', model_dir='outputs/exported/'+name+'_arc/',types=['png','jpg','jepg','png', 'bmp'])


# In[26]:


make_model(embeddings_path='outputs/exported/'+name+'_arc/db.pkl', output_path='outputs/exported/'+name+'_arc/')


# In[29]:


embedding_model = keras.models.load_model('outputs/exported/'+name+'_arc/')
model = load_pkl('outputs/exported/'+name+'_arc/model.pkl')
samples = glob('datasets/processed_merge/*/*.jpg')


# In[36]:


img_paths = samples[0:11]
imgs = [cv2.imread(img).reshape(-1, 128, 128, 3) for img in img_paths]
embedding = [embedding_model.predict(img)[0] for img in imgs]

[print(np.format_float_positional((cosine_similarity(i, embedding[:11])), precision=3), end='\t') for i in embedding]
print()
[print(i, end='\t') for i in model.predict(embedding)]
print()
[print(img.split("/")[-2], end='\t') for img in img_paths] 
print()


# In[ ]:




