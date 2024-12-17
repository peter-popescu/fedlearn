import tensorflow as tf
import tensorflow_datasets as tfds
from keras import backend as K 
import sys
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds


def init():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    return model

def scale_model_weights(weight, scalar):
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final

def sum_scaled_weights(scaled_weight_list):
    avg_grad = list()
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
        
    return avg_grad

def preprocess():
    (ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
    )

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
    return (ds_train, ds_test)

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

def test_model(model, ds_test, i):
    test_size = len(list(ds_test)) // 10
    
    test_slice = ds_test.skip(i * test_size).take(test_size)

    loss, accuracy = model.evaluate(test_slice, verbose=0)
    
    print(f'Loss: {loss}. Acc: {accuracy}')


def aggregate_local_models(client_weights, total_n_examples):

    print("aggregating with", total_n_examples, "samples")
    
    #initial list to collect local model weights after scaling
    scaled_local_weight_list = list()
    for weights, n_examples in client_weights:

        scaling_factor = n_examples / total_n_examples
        scaled_weights = scale_model_weights(weights, scaling_factor)
        scaled_local_weight_list.append(scaled_weights)
        
        K.clear_session()
        
    average_weights = sum_scaled_weights(scaled_local_weight_list)

    return average_weights

def get_weights():
    model = tf.keras.models.load_model('global_model.keras')
    print(len(model.get_weights()))
    return model.get_weights()
