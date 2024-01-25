import tvm
from tvm import te
import numpy



# Define the shape of the tensor
def define_computation(M, N, K):

    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    
    C = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="C")
    
    return A, B, C

def define_computation2(M,N,K):
    bn=32 
    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    
    packedB = te.compute(
                (N / bn, K, bn), lambda bigN, k, littleN: B[k, bigN * bn + littleN], name="packedB"
                )
    C = te.compute(
                (M, N),
                lambda m, n: te.sum(A[m, k] * packedB[n // bn, k, tvm.tir.indexmod(n, bn)], axis=k),
                name="C",
                )
    return A,B,C, packedB


def print_IR(s,A,B,C,name):

    f = tvm.lower(s, [A, B, C], name=name, simple_mode=False)
    print(f)

def build_function(s,A,B,C,name, target):
    mod = tvm.build(s, [A, B, C], target=target, name=name)
    return mod

def evaluate_func(mod,m,n,k,dtype,dev):
    
    input_A =  tvm.nd.array(numpy.random.rand(m,k).astype(dtype), dev)
    input_B =  tvm.nd.array(numpy.random.rand(k,n).astype(dtype), dev)

    # Create an output array to store the result
    output_data = tvm.nd.empty((m,n))

    # Execute the compiled function
    mod(input_A, input_B,  output_data)

    evaluator = mod.time_evaluator(mod.entry_name, dev, number=10,  min_repeat_ms=2000)
    mean_time = evaluator(input_A, input_B, output_data).mean
    gflops=(2.0*m*n*k)/mean_time/1e9
    print("Time: {} GFLOPS: {}".format( mean_time, gflops))


def gemm_v1(m,n,k, dtype, target, dev):

    A,B,C =define_computation(m,n,k)
    
    # Schedule the computation
    s = te.create_schedule(C.op)
    print_IR(s,A,B,C,"gemm_cpu_v1")
    mod = build_function(s,A,B,C,"gemm_cpu_v1",target)
    mod = evaluate_func(mod,m,n,k,dtype,dev)

def gemm_v2(m,n,k, dtype, target, dev):

    A,B,C =define_computation(m,n,k)
    
    # Schedule the computation
    s = te.create_schedule(C.op)
    
    # Blocking by loop tiling
    bn = 32
    kfactor=4
    mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    (kaxis,) = s[C].op.reduce_axis
    ko, ki = s[C].split(kaxis, factor=kfactor)

    # Hoist reduction domain outside the blocking loop
    s[C].reorder(mo, no, ko, ki, mi, ni)


    print_IR(s,A,B,C,"gemm_cpu_v2")
    mod = build_function(s,A,B,C,"gemm_cpu_v2",target)
    mod = evaluate_func(mod,m,n,k,dtype,dev)

def gemm_v3(m,n,k, dtype, target, dev):

    A,B,C =define_computation(m,n,k)

    # Schedule the computation
    s = te.create_schedule(C.op)
    
    # Blocking by loop tiling
    bn = 32
    kfactor=4
    mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    (kaxis,) = s[C].op.reduce_axis
    ko, ki = s[C].split(kaxis, factor=kfactor)

    # Hoist reduction domain outside the blocking loop
    
    s[C].reorder(mo, no, ko, mi, ki, ni)    
    s[C].vectorize(ni)

    print_IR(s,A,B,C,"gemm_cpu_v3")
    mod = build_function(s,A,B,C,"gemm_cpu_v3",target)
    mod = evaluate_func(mod,m,n,k,dtype,dev)


def gemm_v4(m,n,k, dtype, target, dev):

    A,B,C, packedB = define_computation2(m,n,k)

    # Schedule the computation
    s = te.create_schedule(C.op)
    
    # Blocking by loop tiling
    bn = 32
    kfactor=4
    mo, no, mi, ni = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
    (kaxis,) = s[C].op.reduce_axis
    ko, ki = s[C].split(kaxis, factor=kfactor)

    # Hoist reduction domain outside the blocking loop
    
    s[C].reorder(mo, no, ko, mi, ki, ni)    
    s[C].vectorize(ni)
    s[C].parallel(mo)

    bigN, _, littleN = s[packedB].op.axis
    s[packedB].vectorize(littleN)
    s[packedB].parallel(bigN)
    
    print_IR(s,A,B,C,"gemm_cpu_v4")
    mod = build_function(s,A,B,C,"gemm_cpu_v4",target)
    mod = evaluate_func(mod,m,n,k,dtype,dev)


dtype = "float32"

"""
A few examples for specific targets :
 - ARMv8a (8 .2 ) with NEON support
     llvm - device = arm_cpu - mattr =+ v8.2a ,+ fp - armv8 ,+ neon "
 - ARMv8a (8 .2 ) with NEON and FP16 support
     llvm - device = arm_cpu - mattr =+ v8.2a ,+ fp - armv8 ,+ neon ,+ fp16fml
 - AMD Zen2 with AVX2 support
     llvm - mcpu = znver2
 - Intel Icelake with AVX512 support
     llvm - mcpu = icelake - server
"""
target = "llvm"
target = "llvm -device=arm_cpu -mattr=+v8.2a,+fp-armv8,+neon"
dev = tvm.device(target, 0)
m = n = k = 1024
print("######### V1 ############")
gemm_v1(m,n,k,dtype,target,dev)
print("PRESS ANY KEY")
input1 = input()
print("######### V2 ############")
gemm_v2(m,n,k,dtype,target,dev)
print("PRESS ANY KEY")
input1 = input()
print("######### V3 ############")
gemm_v3(m,n,k,dtype,target,dev)
print("PRESS ANY KEY")
input1 = input()
print("######### V4 ############")
gemm_v4(m,n,k,dtype,target,dev)

