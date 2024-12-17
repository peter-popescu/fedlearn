package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"time"

	aggregator "fedlearn/pkg/aggregator"
	"fedlearn/pkg/modelpb"
	"fedlearn/pkg/pb"

	"golang.org/x/exp/rand"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type Server struct {
	// some kind of state struct?
	ModelState aggregator.ModelInfo

	pb.UnimplementedFedLearnServer

	client modelpb.ModelServiceClient
}

func main() {
	if len(os.Args) < 2 {
		log.Fatalf("usage:  %s <tcpport> <file0> [file 1] [file 2] ...",
			os.Args[0])
	}

	port := os.Args[1]

	listen_addr := fmt.Sprintf(":%s", port)
	addr, err := net.ResolveTCPAddr("tcp4", listen_addr)
	if err != nil {
		log.Panicln("error translating address: ", err)
	}

	conn, err := net.ListenTCP("tcp4", addr)
	if err != nil {
		log.Panicln(err)
	}
	fmt.Println("listening")

	defer conn.Close()

	server := &Server{
		ModelState: aggregator.ModelInfo{ClientMap: make(map[uint32]*aggregator.ClientInfo)},
	}

	go func() {
		s := grpc.NewServer()
		pb.RegisterFedLearnServer(s, server)
		if err := s.Serve(conn); err != nil {
			log.Fatal("Failed to serve", err)
		}
	}()

	conn2, err := grpc.NewClient("localhost:999", grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		log.Fatalf("failed to connect: %v", err)
	}
	defer conn2.Close()
	fmt.Println("connectd")

	client := modelpb.NewModelServiceClient(conn2)
	server.client = client
	res, err := client.InitializeModel(context.Background(), &modelpb.InitializeModelReq{})
	if err != nil {
		log.Fatalf("failed to initialize: %v", err)
	}
	log.Printf("book list: %v", res)
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			server.aggregateModelWeights()
		}
	}
}

func (s *Server) aggregateModelWeights() {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	stream, err := s.client.AggregateModelWeights(ctx)
	if err != nil {
		log.Fatalf("client. failed: %v", err)
	}
	for _, client := range s.ModelState.ClientMap {
		if len(client.LocalWeights) <= 0 {
			continue
		}
		if !client.Updated { // don't send if not updated
			continue
		}

		if err := stream.Send(&modelpb.ClientWeights{Weights: client.LocalWeights, ClientDataSize: client.AmountOfData}); err != nil {
			log.Fatalf("client.RecordRoute: stream.Send(%v) failed: %v", client, err)
		}

		client.Updated = false // reset after sending data
	}

	_, err = stream.CloseAndRecv()
	if err != nil {
		log.Fatalf("client.RecordRoute failed: %v", err)
	}

	// message AggregateModelWeightsReq {
	// 	map<uint32, bytes> pairs = 1;
	// }

	// _, err := s.client.AggregateModelWeights(context.Background(), &modelpb.AggregateModelWeightsReq{Pairs: make(map[uint32][]byte)})
	// if err != nil {
	// 	log.Fatalf("")
	// }
	// fmt.Println(reply)
}

/*
// runRecordRoute sends a sequence of points to server and expects to get a RouteSummary from server.
func runRecordRoute(client pb.RouteGuideClient) {
	// Create a random number of random points
	pointCount := int(rand.Int32N(100)) + 2 // Traverse at least two points
	var points []*pb.Point
	for i := 0; i < pointCount; i++ {
		points = append(points, randomPoint())
	}
	log.Printf("Traversing %d points.", len(points))
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	stream, err := client.RecordRoute(ctx)
	if err != nil {
		log.Fatalf("client.RecordRoute failed: %v", err)
	}
	for _, point := range points {
		if err := stream.Send(point); err != nil {
			log.Fatalf("client.RecordRoute: stream.Send(%v) failed: %v", point, err)
		}
	}
	reply, err := stream.CloseAndRecv()
	if err != nil {
		log.Fatalf("client.RecordRoute failed: %v", err)
	}
	log.Printf("Route summary: %v", reply)
}
*/

func (s *Server) SendWeights(ctx context.Context, in *pb.SendWeightsReq) (*pb.SendWeightsRes, error) {
	log.Println("Received weights", len(in.WeightsData))
	s.ModelState.ClientMap[in.ClientId].LocalWeights = in.WeightsData
	s.ModelState.ClientMap[in.ClientId].Updated = true
	s.ModelState.ClientMap[in.ClientId].AmountOfData = in.ClientDataSize

	// TODO: result := <- do something with the weights

	return &pb.SendWeightsRes{Status: "Worked"}, nil
}

func (s *Server) RequestWeights(ctx context.Context, in *pb.RequestWeightsReq) (*pb.RequestWeightsRes, error) {
	log.Printf("Client requesting weights")

	returnedClientId := in.ClientId
	if in.ClientId == 0 {
		returnedClientId = uint32(rand.Intn(500) + 1)
		s.ModelState.ClientMap[returnedClientId] = &aggregator.ClientInfo{}
	}

	fmt.Println(s.client)
	weights, err := s.client.ModelGetWeights(ctx, &modelpb.ModelGetWeightsReq{})
	if err != nil {
		return nil, err
	}

	return &pb.RequestWeightsRes{WeightsData: weights.Weights, ClientId: returnedClientId}, nil
}
