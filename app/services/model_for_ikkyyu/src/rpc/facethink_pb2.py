# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: facethink.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='facethink.proto',
  package='facethink',
  syntax='proto2',
  serialized_pb=_b('\n\x0f\x66\x61\x63\x65think.proto\x12\tfacethink\"B\n\tDataProto\x12\x11\n\tdata_json\x18\x01 \x01(\t\x12\x0e\n\x06sdk_id\x18\x02 \x01(\t\x12\x12\n\nmethrod_id\x18\x03 \x01(\t\"C\n\nModelProto\x12\x11\n\tdata_json\x18\x01 \x01(\t\x12\x0e\n\x06sdk_id\x18\x02 \x01(\t\x12\x12\n\nmethrod_id\x18\x03 \x01(\t2F\n\ngrpcServer\x12\x38\n\x07process\x12\x14.facethink.DataProto\x1a\x15.facethink.ModelProto\"\x00')
)




_DATAPROTO = _descriptor.Descriptor(
  name='DataProto',
  full_name='facethink.DataProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data_json', full_name='facethink.DataProto.data_json', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sdk_id', full_name='facethink.DataProto.sdk_id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='methrod_id', full_name='facethink.DataProto.methrod_id', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=30,
  serialized_end=96,
)


_MODELPROTO = _descriptor.Descriptor(
  name='ModelProto',
  full_name='facethink.ModelProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data_json', full_name='facethink.ModelProto.data_json', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='sdk_id', full_name='facethink.ModelProto.sdk_id', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='methrod_id', full_name='facethink.ModelProto.methrod_id', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=98,
  serialized_end=165,
)

DESCRIPTOR.message_types_by_name['DataProto'] = _DATAPROTO
DESCRIPTOR.message_types_by_name['ModelProto'] = _MODELPROTO
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

DataProto = _reflection.GeneratedProtocolMessageType('DataProto', (_message.Message,), dict(
  DESCRIPTOR = _DATAPROTO,
  __module__ = 'facethink_pb2'
  # @@protoc_insertion_point(class_scope:facethink.DataProto)
  ))
_sym_db.RegisterMessage(DataProto)

ModelProto = _reflection.GeneratedProtocolMessageType('ModelProto', (_message.Message,), dict(
  DESCRIPTOR = _MODELPROTO,
  __module__ = 'facethink_pb2'
  # @@protoc_insertion_point(class_scope:facethink.ModelProto)
  ))
_sym_db.RegisterMessage(ModelProto)



_GRPCSERVER = _descriptor.ServiceDescriptor(
  name='grpcServer',
  full_name='facethink.grpcServer',
  file=DESCRIPTOR,
  index=0,
  options=None,
  serialized_start=167,
  serialized_end=237,
  methods=[
  _descriptor.MethodDescriptor(
    name='process',
    full_name='facethink.grpcServer.process',
    index=0,
    containing_service=None,
    input_type=_DATAPROTO,
    output_type=_MODELPROTO,
    options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_GRPCSERVER)

DESCRIPTOR.services_by_name['grpcServer'] = _GRPCSERVER

# @@protoc_insertion_point(module_scope)
