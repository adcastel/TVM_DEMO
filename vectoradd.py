import tvm
from tvm import te
import numpy



# Define the shape of the tensor
n = 1000#te.var("n")

def define_computation(n):
    A = te.placeholder((n,), name="A")
    B = te.placeholder((n,), name="B")

    C = te.compute(A.shape, lambda i: A[i] * B[i], name="C")

    return A, B, C

def print_IR(s,A,B,C,name):

    f = tvm.lower(s, [A, B, C], name=name, simple_mode=False)
    print(f)

def build_function(s,A,B,C,name, target):
    mod = tvm.build(s, [A, B, C], target=target, name=name)
    return mod

def evaluate_func(mod,n,dtype,dev):
    N=n
    input_A =  tvm.nd.array(numpy.random.rand(N, ).astype(dtype), dev)
    input_B =  tvm.nd.array(numpy.random.rand(N, ).astype(dtype), dev)

    # Create an output array to store the result
    output_data = tvm.nd.empty(input_A.shape)

    # Execute the compiled function
    mod(input_A, input_B,  output_data)

    evaluator = mod.time_evaluator(mod.entry_name, dev, number=100,  min_repeat_ms=500)
    mean_time = evaluator(input_A, input_B, output_data).mean
    print("Time:", mean_time)


def vector_add_v1(n, dtype, target, dev):

    A,B,C =define_computation(n)
# Schedule the computation
    s = te.create_schedule(C.op)
    print_IR(s,A,B,C,"vectoradd_v1")
    mod = build_function(s,A,B,C,"vectoradd_v1",target)
    mod = evaluate_func(mod,n,dtype,dev)

def vector_add_v2(n, dtype, target, dev):

    A,B,C =define_computation(n)
    s = te.create_schedule(C.op)
    ii, = s[C].op.axis
    io,ii = s[C].split(ii,factor=4)
    print_IR(s,A,B,C,"vectoradd_v2")
    mod = build_function(s,A,B,C,"vectoradd_v2",target)
    mod = evaluate_func(mod,n,dtype,dev)

def vector_add_v3(n, dtype, target, dev):

    A,B,C =define_computation(n)
    s = te.create_schedule(C.op)
    i, = s[C].op.axis
    io,ii = s[C].split(i,factor=4)
    s[C].vectorize(ii)
    print_IR(s,A,B,C,"vectoradd_v3")
    mod = build_function(s,A,B,C,"vectoradd_v3",target)
    mod = evaluate_func(mod,n,dtype,dev)

dtype = "float32"
target = "llvm -device=arm_cpu -mattr=+v8.2a,+fp-armv8,+neon"
dev = tvm.device(target, 0)
n = 2000
print("######### V1 ############")
vector_add_v1(n,dtype,target,dev)
print("######### V2 ############")
vector_add_v2(n,dtype,target,dev)
print("######### V3 ############")
vector_add_v3(n,dtype,target,dev)
# Create input data (e.g., an array of 5 elements)
# Print the result

