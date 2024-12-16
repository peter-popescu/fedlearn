build:
	go build -o out cmd/server/server.go

run:
	./out

runbuild:
	go run cmd/server/server.go 1000

protobuild:
	protoc --go_out=. --go-grpc_out=. proto/fedlearn.proto
	protoc --go_out=. --go-grpc_out=. proto/model.proto

	python3 -m grpc_tools.protoc -I. --python_out=cmd/server/model --grpc_python_out=cmd/server/model proto/model.proto
	python3 -m grpc_tools.protoc -I. --python_out=client --grpc_python_out=client proto/fedlearn.proto