import sys
import logging
import argparse
import ClientModel as cm
import grpc
import proto.fedlearn_pb2     as pb2_fedlearn
import proto.fedlearn_pb2_grpc as pb2_grpc
import pickle
import tensorflow_datasets as tfds

def serialize(model):
    return pickle.dumps({
        "weights": model.get_weights(),
    })

def deserialize(cnn_bytes):
    loaded = pickle.loads(cnn_bytes)
    weights = loaded['weights']
    return weights

def main(input_args):
    parser = argparse.ArgumentParser()
    parser.add_argument("port")

    args = parser.parse_args(input_args)

    addr = "localhost:{}".format(args.port)
    print("Connecting")
    with grpc.insecure_channel(addr) as channel:
        stub = pb2_grpc.FedLearnStub(channel)
        
        train, test = cm.preprocess()

        response = stub.RequestWeights(pb2_fedlearn.RequestWeightsReq(client_id=0, client_data_size=len(train)))
        # print("Received  {}".format(pb2_fedlearn.GuessStatus.Name(response.result)))

        # print(response.weights_data)
        model = cm.init() 
        weights = deserialize(response.weights_data)
        model.set_weights(weights)

        client_id = response.client_id
        # train
        i = 0
        for i in range(10):
            cm.train_client_model(model, train, test, i)
            client_weights = serialize(model)
            print("length", len(client_weights))
            response = stub.SendWeights(pb2_fedlearn.SendWeightsReq(weights_data=client_weights, client_id=client_id, client_data_size=len(train)))
            if response.status != "Worked":
                print("error sending weights:", response.Status)
            else:
                response = stub.RequestWeights(pb2_fedlearn.RequestWeightsReq(client_id=client_id, client_data_size=len(train)))
                weights = deserialize(response.weights_data)
                model.set_weights(weights)

if __name__ == "__main__":
    main(sys.argv[1:])
