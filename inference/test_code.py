import control_message_pb2
from google.protobuf.json_format import MessageToDict

with open('control_message.bin', 'rb') as f:
    serialized_data = f.read()

control_message = control_message_pb2.ControlMessage()
control_message.ParseFromString(serialized_data)

print("Deserialized Control Message:", MessageToDict(control_message, always_print_fields_with_no_presence=True))