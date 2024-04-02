#!/usr/bin/python3

# TVM_LOG_DEBUG=1 python3 -u main.py 2>&1 | tee dump.log

# TVM
import tvm
from tvm import relay
from tvm.contrib.download import download_testdata

# Others
import numpy as np
from sklearn import datasets            # type: ignore

# PyTorch imports
import torch
from torch import nn
from torch import Tensor

# custom simple model
class NN(nn.Module):
    def __init__(self, input_size: int, H1: int, output_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, H1)
        self.linear2 = nn.Linear(H1, output_size)
    
    def forward(self, x: 'Tensor') -> 'Tensor':
        x = torch.sigmoid(self.linear(x))
        x = torch.sigmoid(self.linear2(x))
        return x

model = NN(2, 10, 1)

# get scripted model using tracing
input_shape = [1000, 2]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

# get data
x, y = datasets.make_circles(n_samples=1000, random_state=42, noise=0.04)
x_data = torch.FloatTensor(x)

# import the graph to relay
input_name = "input0"
assert input_shape == list(x_data.numpy().shape)
shape_list = [(input_name, x_data.numpy().shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

# relay build
target = tvm.target.Target("llvm", host="llvm")
dev = tvm.cpu(0)
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

# execute the portable graph on TVM
from tvm.contrib import graph_executor

dtype = "float32"
m = graph_executor.GraphModule(lib["default"](dev))
m.set_input(input_name, tvm.nd.array(x_data.numpy().astype(dtype)))

# Execute
m.run()

# Get outputs
tvm_output = m.get_output(0)

# Get top-1 result for TVM
top1_tvm = np.argmax(tvm_output.numpy())

# Convert input to PyTorch variable and get PyTorch result for comparison
with torch.no_grad():
    output = model(x_data)

    # Get top-1 result for PyTorch
    top1_torch = np.argmax(output.numpy())

print("Relay top-1 id: {}".format(top1_tvm))
print("Torch top-1 id: {}".format(top1_torch))
