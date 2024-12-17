package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"sync"
	"time"

	aggregator "fedlearn/pkg/aggregator"
	"fedlearn/pkg/modelpb"
	"fedlearn/pkg/pb"

	"golang.org/x/exp/rand"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type Server struct {
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

	var mu sync.Mutex
	server := &Server{
		ModelState: aggregator.ModelInfo{ClientMap: make(map[uint32]*aggregator.ClientInfo), Mu: &mu},
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
			go server.aggregateModelWeights()
		}
	}
}

func (s *Server) aggregateModelWeights() {
	if len(s.ModelState.ClientMap) == 0 {
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	s.ModelState.Mu.Lock()

	fedAvgClientAmount := 0.5

	count := 0
	for _, client := range s.ModelState.ClientMap {
		if client.Updated {
			count += 1
		}
	}

	amount := float64(count) / float64(len(s.ModelState.ClientMap))
	if amount < fedAvgClientAmount {
		s.ModelState.Mu.Unlock()
		return
	}

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
			s.ModelState.Mu.Unlock()
			log.Fatalf("client.RecordRoute: stream.Send(%v) failed: %v", client, err)
		}

		client.Updated = false // reset after sending data
	}

	s.ModelState.Mu.Unlock()

	_, err = stream.CloseAndRecv()
	if err != nil {
		s.ModelState.Mu.Unlock()
		log.Fatalf("client.RecordRoute failed: %v", err)
	}

	s.client.TestModel(ctx, &modelpb.TestModelReq{})
}

func (s *Server) SendWeights(ctx context.Context, in *pb.SendWeightsReq) (*pb.SendWeightsRes, error) {
	s.ModelState.Mu.Lock()
	s.ModelState.ClientMap[in.ClientId].LocalWeights = in.WeightsData
	s.ModelState.ClientMap[in.ClientId].Updated = true
	s.ModelState.ClientMap[in.ClientId].AmountOfData = in.ClientDataSize
	s.ModelState.Mu.Unlock()

	return &pb.SendWeightsRes{Status: "Worked"}, nil
}

func (s *Server) RequestWeights(ctx context.Context, in *pb.RequestWeightsReq) (*pb.RequestWeightsRes, error) {

	s.ModelState.Mu.Lock()
	defer s.ModelState.Mu.Unlock()

	returnedClientId := in.ClientId
	if in.ClientId == 0 {
		returnedClientId = uint32(rand.Intn(500) + 1)
		s.ModelState.ClientMap[returnedClientId] = &aggregator.ClientInfo{}
	}

	weights, err := s.client.ModelGetWeights(ctx, &modelpb.ModelGetWeightsReq{})
	if err != nil {
		return nil, err
	}

	return &pb.RequestWeightsRes{WeightsData: weights.Weights, ClientId: returnedClientId}, nil
}
