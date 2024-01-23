import tvm
from tvm import te
import numpy

dtype = "float32"
target = "llvm -device=arm_cpu -mattr=+v8.2a,+fp-armv8,+neon"
dev = tvm.device(target, 0)


# Define the shape of the tensor
n = te.var("n")
A = te.placeholder((n,), name="A")

# Define the computation (Hello World in the context of TVM)
B = te.compute(A.shape, lambda i: A[i] + 1, name="B")

# Schedule the computation
s = te.create_schedule(B.op)


#f = tvm.lower(s, [A, B], name="helloworld", simple_mode=False)

# Compile the function
mod = tvm.build(s, [A, B])

# Create input data (e.g., an array of 5 elements)
input_data =  tvm.nd.array(numpy.random.rand(5, ).astype(dtype), dev)

# Create an output array to store the result
output_data = tvm.nd.empty(input_data.shape)

# Execute the compiled function
mod(input_data, output_data)

# Print the result
print("Input:", input_data.asnumpy())
print("Output:", output_data.asnumpy())

