import tensorflow as tf
from Model import init, aggregate_local_models, get_weights
import proto.model_pb2     as pb2_model
import proto.model_pb2_grpc as pb2_grpc
import pickle
import grpc
from concurrent import futures
import tensorflow_datasets as tfds

def serialize(model):
    return pickle.dumps({
        "weights": model.get_weights(),
    })


def deserialize(cnn_bytes):
    loaded = pickle.loads(cnn_bytes)
    weights = loaded['weights']
    return weights

class ModelServiceServicer(pb2_grpc.ModelServiceServicer):
    def __init__(self):
       self.model = None
       self.test_data = preprocess()
       
    def InitializeModel(self, request, context):
        if self.model==None:
            self.model = init()
            return pb2_model.InitializeModelRes(status=0)
        return pb2_model.InitializeModelRes(status=2)

    def ModelGetWeights(self, request, context):
        serialized_weights = serialize(self.model)
        return pb2_model.ClientWeights(weights=serialized_weights)

    def AggregateModelWeights(self, request_iterator, context):
        client_weights = []
        total_n_examples = 0
        for client in request_iterator:
            dslzd_weights = deserialize(client.weights)
            n_examples = client.client_data_size
            total_n_examples += n_examples
            client_weights.append((dslzd_weights, n_examples))
        
        if len(client_weights) == 0:
            return pb2_model.AggregateModelWeightsRes(status=0)
        new_weights = aggregate_local_models(client_weights, total_n_examples)
        self.model.set_weights(new_weights)

        return pb2_model.AggregateModelWeightsRes(status=0)
    
    def TestModel(self, req, context):
        print("testing")
        if self.model:
            loss, accuracy = self.model.evaluate(self.test_data, verbose=0)
            print(f'loss: {loss}, acc: {accuracy}')
            
            return pb2_model.TestModelRes(loss=loss, acc=accuracy)
        return pb2_model.TestModelRes(loss=0, acc=0)
            
def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

def preprocess():
    (ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
    )

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
    return ds_test
            
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    pb2_grpc.add_ModelServiceServicer_to_server(
        ModelServiceServicer(), server
    )
    server.add_insecure_port("[::]:999")
    server.start()
    server.wait_for_termination()



if __name__ == "__main__":
    serve()