package aggregator

type ModelInfo struct {
	ClientMap map[uint32]*ClientInfo
}

type ClientInfo struct {
	LocalWeights []byte
	Aggregated   bool
	ClientId     uint32
	AmountOfData uint32
}

func IntializeAggregator() *ModelInfo {
	return &ModelInfo{}
}
