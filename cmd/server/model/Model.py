import tensorflow as tf
import tensorflow_datasets as tfds
from keras import backend as K 
import sys
import argparse


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

    # model.save("global_model.keras")
    return model

# def weight_scalling_factor(clients_trn_data, client_name):
#     client_names = list(clients_trn_data.keys())
#     bs = list(clients_trn_data[client_name])[0][0].shape[0]
#     global_count = sum([tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy() for client_name in client_names])*bs
#     local_count = tf.data.experimental.cardinality(clients_trn_data[client_name]).numpy()*bs
#     return local_count/global_count

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

def aggregate_local_models(client_weights, total_n_examples):
    
    scaled_local_weight_list = list()
    for weights, n_examples in client_weights:
        #initial list to collect local model weights after scalling

        # scaling_factor = weight_scalling_factor(clients_batched, client)
        scaling_factor = n_examples / total_n_examples
        scaled_weights = scale_model_weights(weights, scaling_factor)
        scaled_local_weight_list.append(scaled_weights)
        
        K.clear_session()
        
    average_weights = sum_scaled_weights(scaled_local_weight_list)

    # TODO: should have testing data on the server side for metrics
    # for(X_test, Y_test) in test_batched:
    #     global_acc, global_loss = test_model(X_test, Y_test, global_model, comm_round)

    return average_weights

def get_weights():
    model = tf.keras.models.load_model('global_model.keras')
    print(len(model.get_weights()))
    return model.get_weights()

# def main(args):
#     print("call")
#     parser = argparse.ArgumentParser()
#     parser.add_argument("function")
#     parser.add_argument("-c", "--client_info")
#     args = parser.parse_args(args)
#     function = args.function
#     match function:
#         case "init":
#             init()
#         case "aggregate_local_models":
#             client_info = args.client_info
#             aggregate_local_models(client_info)
#         case "get_weights":
#             get_weights()

# if __name__ == "__main__":
#     main(sys.argv[1:])