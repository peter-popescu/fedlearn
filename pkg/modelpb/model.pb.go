// Code generated by protoc-gen-go. DO NOT EDIT.
// source: proto/model.proto

package modelpb

import (
	fmt "fmt"
	proto "github.com/golang/protobuf/proto"
	math "math"
)

// Reference imports to suppress errors if they are not otherwise used.
var _ = proto.Marshal
var _ = fmt.Errorf
var _ = math.Inf

// This is a compile-time assertion to ensure that this generated file
// is compatible with the proto package it is being compiled against.
// A compilation error at this line likely means your copy of the
// proto package needs to be updated.
const _ = proto.ProtoPackageIsVersion3 // please upgrade the proto package

type ModelGetWeightsReq struct {
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *ModelGetWeightsReq) Reset()         { *m = ModelGetWeightsReq{} }
func (m *ModelGetWeightsReq) String() string { return proto.CompactTextString(m) }
func (*ModelGetWeightsReq) ProtoMessage()    {}
func (*ModelGetWeightsReq) Descriptor() ([]byte, []int) {
	return fileDescriptor_d10048d769ba1c4e, []int{0}
}

func (m *ModelGetWeightsReq) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_ModelGetWeightsReq.Unmarshal(m, b)
}
func (m *ModelGetWeightsReq) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_ModelGetWeightsReq.Marshal(b, m, deterministic)
}
func (m *ModelGetWeightsReq) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ModelGetWeightsReq.Merge(m, src)
}
func (m *ModelGetWeightsReq) XXX_Size() int {
	return xxx_messageInfo_ModelGetWeightsReq.Size(m)
}
func (m *ModelGetWeightsReq) XXX_DiscardUnknown() {
	xxx_messageInfo_ModelGetWeightsReq.DiscardUnknown(m)
}

var xxx_messageInfo_ModelGetWeightsReq proto.InternalMessageInfo

type InitializeModelReq struct {
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *InitializeModelReq) Reset()         { *m = InitializeModelReq{} }
func (m *InitializeModelReq) String() string { return proto.CompactTextString(m) }
func (*InitializeModelReq) ProtoMessage()    {}
func (*InitializeModelReq) Descriptor() ([]byte, []int) {
	return fileDescriptor_d10048d769ba1c4e, []int{1}
}

func (m *InitializeModelReq) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_InitializeModelReq.Unmarshal(m, b)
}
func (m *InitializeModelReq) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_InitializeModelReq.Marshal(b, m, deterministic)
}
func (m *InitializeModelReq) XXX_Merge(src proto.Message) {
	xxx_messageInfo_InitializeModelReq.Merge(m, src)
}
func (m *InitializeModelReq) XXX_Size() int {
	return xxx_messageInfo_InitializeModelReq.Size(m)
}
func (m *InitializeModelReq) XXX_DiscardUnknown() {
	xxx_messageInfo_InitializeModelReq.DiscardUnknown(m)
}

var xxx_messageInfo_InitializeModelReq proto.InternalMessageInfo

type InitializeModelRes struct {
	Status               uint32   `protobuf:"varint,1,opt,name=status,proto3" json:"status,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *InitializeModelRes) Reset()         { *m = InitializeModelRes{} }
func (m *InitializeModelRes) String() string { return proto.CompactTextString(m) }
func (*InitializeModelRes) ProtoMessage()    {}
func (*InitializeModelRes) Descriptor() ([]byte, []int) {
	return fileDescriptor_d10048d769ba1c4e, []int{2}
}

func (m *InitializeModelRes) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_InitializeModelRes.Unmarshal(m, b)
}
func (m *InitializeModelRes) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_InitializeModelRes.Marshal(b, m, deterministic)
}
func (m *InitializeModelRes) XXX_Merge(src proto.Message) {
	xxx_messageInfo_InitializeModelRes.Merge(m, src)
}
func (m *InitializeModelRes) XXX_Size() int {
	return xxx_messageInfo_InitializeModelRes.Size(m)
}
func (m *InitializeModelRes) XXX_DiscardUnknown() {
	xxx_messageInfo_InitializeModelRes.DiscardUnknown(m)
}

var xxx_messageInfo_InitializeModelRes proto.InternalMessageInfo

func (m *InitializeModelRes) GetStatus() uint32 {
	if m != nil {
		return m.Status
	}
	return 0
}

type ClientWeights struct {
	Weights              []byte   `protobuf:"bytes,1,opt,name=weights,proto3" json:"weights,omitempty"`
	ClientDataSize       uint32   `protobuf:"varint,2,opt,name=client_data_size,json=clientDataSize,proto3" json:"client_data_size,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *ClientWeights) Reset()         { *m = ClientWeights{} }
func (m *ClientWeights) String() string { return proto.CompactTextString(m) }
func (*ClientWeights) ProtoMessage()    {}
func (*ClientWeights) Descriptor() ([]byte, []int) {
	return fileDescriptor_d10048d769ba1c4e, []int{3}
}

func (m *ClientWeights) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_ClientWeights.Unmarshal(m, b)
}
func (m *ClientWeights) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_ClientWeights.Marshal(b, m, deterministic)
}
func (m *ClientWeights) XXX_Merge(src proto.Message) {
	xxx_messageInfo_ClientWeights.Merge(m, src)
}
func (m *ClientWeights) XXX_Size() int {
	return xxx_messageInfo_ClientWeights.Size(m)
}
func (m *ClientWeights) XXX_DiscardUnknown() {
	xxx_messageInfo_ClientWeights.DiscardUnknown(m)
}

var xxx_messageInfo_ClientWeights proto.InternalMessageInfo

func (m *ClientWeights) GetWeights() []byte {
	if m != nil {
		return m.Weights
	}
	return nil
}

func (m *ClientWeights) GetClientDataSize() uint32 {
	if m != nil {
		return m.ClientDataSize
	}
	return 0
}

type Weights struct {
	Weights              []byte   `protobuf:"bytes,1,opt,name=weights,proto3" json:"weights,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *Weights) Reset()         { *m = Weights{} }
func (m *Weights) String() string { return proto.CompactTextString(m) }
func (*Weights) ProtoMessage()    {}
func (*Weights) Descriptor() ([]byte, []int) {
	return fileDescriptor_d10048d769ba1c4e, []int{4}
}

func (m *Weights) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_Weights.Unmarshal(m, b)
}
func (m *Weights) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_Weights.Marshal(b, m, deterministic)
}
func (m *Weights) XXX_Merge(src proto.Message) {
	xxx_messageInfo_Weights.Merge(m, src)
}
func (m *Weights) XXX_Size() int {
	return xxx_messageInfo_Weights.Size(m)
}
func (m *Weights) XXX_DiscardUnknown() {
	xxx_messageInfo_Weights.DiscardUnknown(m)
}

var xxx_messageInfo_Weights proto.InternalMessageInfo

func (m *Weights) GetWeights() []byte {
	if m != nil {
		return m.Weights
	}
	return nil
}

type AggregateModelWeightsRes struct {
	Status               uint32   `protobuf:"varint,1,opt,name=status,proto3" json:"status,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *AggregateModelWeightsRes) Reset()         { *m = AggregateModelWeightsRes{} }
func (m *AggregateModelWeightsRes) String() string { return proto.CompactTextString(m) }
func (*AggregateModelWeightsRes) ProtoMessage()    {}
func (*AggregateModelWeightsRes) Descriptor() ([]byte, []int) {
	return fileDescriptor_d10048d769ba1c4e, []int{5}
}

func (m *AggregateModelWeightsRes) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_AggregateModelWeightsRes.Unmarshal(m, b)
}
func (m *AggregateModelWeightsRes) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_AggregateModelWeightsRes.Marshal(b, m, deterministic)
}
func (m *AggregateModelWeightsRes) XXX_Merge(src proto.Message) {
	xxx_messageInfo_AggregateModelWeightsRes.Merge(m, src)
}
func (m *AggregateModelWeightsRes) XXX_Size() int {
	return xxx_messageInfo_AggregateModelWeightsRes.Size(m)
}
func (m *AggregateModelWeightsRes) XXX_DiscardUnknown() {
	xxx_messageInfo_AggregateModelWeightsRes.DiscardUnknown(m)
}

var xxx_messageInfo_AggregateModelWeightsRes proto.InternalMessageInfo

func (m *AggregateModelWeightsRes) GetStatus() uint32 {
	if m != nil {
		return m.Status
	}
	return 0
}

type TestModelReq struct {
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *TestModelReq) Reset()         { *m = TestModelReq{} }
func (m *TestModelReq) String() string { return proto.CompactTextString(m) }
func (*TestModelReq) ProtoMessage()    {}
func (*TestModelReq) Descriptor() ([]byte, []int) {
	return fileDescriptor_d10048d769ba1c4e, []int{6}
}

func (m *TestModelReq) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_TestModelReq.Unmarshal(m, b)
}
func (m *TestModelReq) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_TestModelReq.Marshal(b, m, deterministic)
}
func (m *TestModelReq) XXX_Merge(src proto.Message) {
	xxx_messageInfo_TestModelReq.Merge(m, src)
}
func (m *TestModelReq) XXX_Size() int {
	return xxx_messageInfo_TestModelReq.Size(m)
}
func (m *TestModelReq) XXX_DiscardUnknown() {
	xxx_messageInfo_TestModelReq.DiscardUnknown(m)
}

var xxx_messageInfo_TestModelReq proto.InternalMessageInfo

type TestModelRes struct {
	Loss                 float32  `protobuf:"fixed32,1,opt,name=loss,proto3" json:"loss,omitempty"`
	Acc                  float32  `protobuf:"fixed32,2,opt,name=acc,proto3" json:"acc,omitempty"`
	XXX_NoUnkeyedLiteral struct{} `json:"-"`
	XXX_unrecognized     []byte   `json:"-"`
	XXX_sizecache        int32    `json:"-"`
}

func (m *TestModelRes) Reset()         { *m = TestModelRes{} }
func (m *TestModelRes) String() string { return proto.CompactTextString(m) }
func (*TestModelRes) ProtoMessage()    {}
func (*TestModelRes) Descriptor() ([]byte, []int) {
	return fileDescriptor_d10048d769ba1c4e, []int{7}
}

func (m *TestModelRes) XXX_Unmarshal(b []byte) error {
	return xxx_messageInfo_TestModelRes.Unmarshal(m, b)
}
func (m *TestModelRes) XXX_Marshal(b []byte, deterministic bool) ([]byte, error) {
	return xxx_messageInfo_TestModelRes.Marshal(b, m, deterministic)
}
func (m *TestModelRes) XXX_Merge(src proto.Message) {
	xxx_messageInfo_TestModelRes.Merge(m, src)
}
func (m *TestModelRes) XXX_Size() int {
	return xxx_messageInfo_TestModelRes.Size(m)
}
func (m *TestModelRes) XXX_DiscardUnknown() {
	xxx_messageInfo_TestModelRes.DiscardUnknown(m)
}

var xxx_messageInfo_TestModelRes proto.InternalMessageInfo

func (m *TestModelRes) GetLoss() float32 {
	if m != nil {
		return m.Loss
	}
	return 0
}

func (m *TestModelRes) GetAcc() float32 {
	if m != nil {
		return m.Acc
	}
	return 0
}

func init() {
	proto.RegisterType((*ModelGetWeightsReq)(nil), "ModelGetWeightsReq")
	proto.RegisterType((*InitializeModelReq)(nil), "InitializeModelReq")
	proto.RegisterType((*InitializeModelRes)(nil), "InitializeModelRes")
	proto.RegisterType((*ClientWeights)(nil), "ClientWeights")
	proto.RegisterType((*Weights)(nil), "Weights")
	proto.RegisterType((*AggregateModelWeightsRes)(nil), "AggregateModelWeightsRes")
	proto.RegisterType((*TestModelReq)(nil), "TestModelReq")
	proto.RegisterType((*TestModelRes)(nil), "TestModelRes")
}

func init() {
	proto.RegisterFile("proto/model.proto", fileDescriptor_d10048d769ba1c4e)
}

var fileDescriptor_d10048d769ba1c4e = []byte{
	// 308 bytes of a gzipped FileDescriptorProto
	0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0x84, 0x52, 0x41, 0x4b, 0xf3, 0x40,
	0x10, 0x25, 0xfd, 0x3e, 0x5a, 0x1c, 0x93, 0xb4, 0x4e, 0x55, 0x62, 0x4e, 0x12, 0x2f, 0x11, 0x64,
	0x85, 0x2a, 0x5e, 0x3c, 0xa9, 0x05, 0xf1, 0xe0, 0x25, 0x11, 0x04, 0x2f, 0x65, 0x9b, 0x0e, 0x71,
	0x31, 0x36, 0x31, 0xbb, 0x2a, 0xe4, 0x77, 0xfb, 0x03, 0x24, 0x1b, 0x1b, 0x4c, 0x9a, 0xe2, 0xed,
	0xbd, 0xb7, 0xb3, 0xc3, 0xbc, 0x37, 0x03, 0x3b, 0x59, 0x9e, 0xaa, 0xf4, 0xf4, 0x35, 0x5d, 0x50,
	0xc2, 0x34, 0xf6, 0x76, 0x01, 0xef, 0x4b, 0x7a, 0x4b, 0xea, 0x91, 0x44, 0xfc, 0xac, 0x64, 0x40,
	0x6f, 0xa5, 0x7a, 0xb7, 0x14, 0x4a, 0xf0, 0x44, 0x14, 0xa4, 0xdf, 0x4b, 0xf5, 0xa4, 0x43, 0x95,
	0xb8, 0x0f, 0x7d, 0xa9, 0xb8, 0x7a, 0x97, 0x8e, 0x71, 0x68, 0xf8, 0x56, 0xf0, 0xc3, 0xbc, 0x10,
	0xac, 0x9b, 0x44, 0xd0, 0x72, 0xd5, 0x17, 0x1d, 0x18, 0x7c, 0x56, 0x50, 0x57, 0x9a, 0xc1, 0x8a,
	0xa2, 0x0f, 0xa3, 0x48, 0x97, 0xce, 0x16, 0x5c, 0xf1, 0x99, 0x14, 0x05, 0x39, 0x3d, 0xdd, 0xcc,
	0xae, 0xf4, 0x29, 0x57, 0x3c, 0x14, 0x05, 0x79, 0x47, 0x30, 0xf8, 0xb3, 0x9d, 0x37, 0x01, 0xe7,
	0x2a, 0x8e, 0x73, 0x8a, 0xb9, 0xaa, 0xc6, 0xac, 0x9d, 0x6d, 0x9e, 0xd6, 0x06, 0xf3, 0x81, 0xa4,
	0xaa, 0xbd, 0x9e, 0x37, 0xb8, 0x44, 0x84, 0xff, 0x49, 0x2a, 0xab, 0x5f, 0xbd, 0x40, 0x63, 0x1c,
	0xc1, 0x3f, 0x1e, 0x45, 0x7a, 0xd2, 0x5e, 0x50, 0xc2, 0xc9, 0x97, 0x01, 0xa6, 0xfe, 0x12, 0x52,
	0xfe, 0x21, 0x22, 0xc2, 0x4b, 0x18, 0xb6, 0x22, 0xc3, 0x31, 0x5b, 0x8f, 0xd6, 0xed, 0x10, 0x25,
	0x5e, 0xc0, 0xb0, 0xb5, 0x1b, 0x1c, 0xb3, 0xf5, 0x6d, 0xb9, 0x36, 0x6b, 0x06, 0x3d, 0x85, 0xbd,
	0x4e, 0xff, 0xd8, 0x2a, 0x74, 0x0f, 0xd8, 0xa6, 0x9c, 0x7c, 0x03, 0x8f, 0x61, 0xab, 0x4e, 0x00,
	0x2d, 0xf6, 0x3b, 0x1d, 0xb7, 0x41, 0xe5, 0xb5, 0xf5, 0xb4, 0x9d, 0xbd, 0xc4, 0xd5, 0x5d, 0x65,
	0xf3, 0x79, 0x5f, 0x9f, 0xd6, 0xd9, 0x77, 0x00, 0x00, 0x00, 0xff, 0xff, 0xc8, 0x83, 0x45, 0x75,
	0x6f, 0x02, 0x00, 0x00,
}
