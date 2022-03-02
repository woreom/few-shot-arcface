"""This module provides the dataset parsing function to generate the training
and testing data."""

import tensorflow as tf
from arcface.preprocessing import normalize
from sklearn.model_selection import train_test_split


def build_dataset(tfrecord_file,
                  batch_size,
                  one_hot_depth,
                  training=False,
                  val_size = 0.0,
                  test_size = 0.0,
                  num_examples = 12000,
                  buffer_size=4096):
    """Generate parsed TensorFlow dataset.

    Args:
        tfrecord_file: the tfrecord file path.
        batch_size: batch size.
        one_hot_depth: the depth for one hot encoding, usually the number of 
            classes.
        training: a boolean indicating whether the dataset will be used for
            training.
        buffer_size: hwo large the buffer is for shuffling the samples.

    Returns:
        a parsed dataset.
    """
    # Let TensorFlow tune the input pipeline automatically.
    autotune = tf.data.experimental.AUTOTUNE

    # Describe how the dataset was constructed. The author who created the file
    # is responsible for this information.
    feature_description = {'image/height': tf.io.FixedLenFeature([], tf.int64),
                           'image/width': tf.io.FixedLenFeature([], tf.int64),
                           'image/depth': tf.io.FixedLenFeature([], tf.int64),
                           'image/encoded': tf.io.FixedLenFeature([], tf.string),
                           'label': tf.io.FixedLenFeature([], tf.int64)}

    # Define a helper function to decode the tf-example. This function will be
    # called by map() later.
    def _parse_function(example):
        features = tf.io.parse_single_example(example, feature_description)
        image = tf.image.decode_jpeg(features['image/encoded'])
        
        # random flip
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        # random rotation
        image = tf.image.rot90(image, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
        
        image = normalize(image)
        

        
        
        label = tf.one_hot(features['label'], depth=one_hot_depth,
                           dtype=tf.float32)

        return image, label
    

    # Now construct the dataset from tfrecord file and make it indefinite.
    dataset = tf.data.TFRecordDataset(tfrecord_file)

    # Shuffle the data if training.
    if training:
        dataset = dataset.shuffle(buffer_size)

    # Parse the dataset to get samples.
    dataset = dataset.map(_parse_function, num_parallel_calls=autotune)

    # Batch the data.
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Prefetch the data to accelerate the pipeline.
    dataset = dataset.prefetch(autotune)
    
    if training:
        if test_size != 0.0:
            test_size = int(test_size * num_examples)

            test_dataset = dataset.take(test_size)
            dataset = dataset.skip(test_size)
        else:
            test_dataset = None

        if test_size != 0.0:
            val_size = int(val_size * (num_examples-test_size))

            val_dataset = dataset.take(val_size)
            dataset = dataset.skip(test_size)
        else:
            val_dataset=None

        return dataset, val_dataset ,test_dataset
    
    return dataset
