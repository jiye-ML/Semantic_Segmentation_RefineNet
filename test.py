from tensorflow.python import pywrap_tensorflow
import os

checkpint_path = "checkpoints/RefineNet_step_100000.ckpt"
reader = pywrap_tensorflow.NewCheckpointReader(checkpint_path)
var_to_shape_map = reader.get_variable_to_shape_map()

for key in var_to_shape_map:
    print("tensor_name: ", key)
    # print(reader.get_tensor(key))