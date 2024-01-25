import tvm
import tvm.testing
from tvm import te
import numpy
import timeit


# The size of the matrix
# (M, K) x (K, N)
# You are free to try out different shapes, sometimes TVM optimization outperforms numpy with MKL.




# You will want to adjust the target to match any CPU vector extensions you
# might have. For example, if you're using using Intel AVX2 (Advanced Vector
# Extensions) ISA for SIMD, you can get the best performance by changing the
# following line to ``llvm -mcpu=core-avx2``, or specific type of CPU you use.
# Recall that you're using llvm, you can get this information from the command
# ``llc --version`` to get the CPU type, and you can check ``/proc/cpuinfo``
# for additional extensions that your processor might support.
# Repeatedly perform a matrix multiplication to get a performance baseline
# for the default numpy implementation

def performance_numpy(M,N,K):
	np_repeat = 100
	np_running_time = timeit.timeit(
	   	setup="import numpy\n"
	   	"M = " + str(M) + "\n"
	   	"K = " + str(K) + "\n"
	   	"N = " + str(N) + "\n"
	   	'dtype = "float32"\n'
	   	"a = numpy.random.rand(M, K).astype(dtype)\n"
	   	"b = numpy.random.rand(K, N).astype(dtype)\n",
	   	stmt="answer = numpy.dot(a, b)",
	   	number=np_repeat,
	)
	ops=2.0*M*N*K/1e9
	t=np_running_time / np_repeat
	gflops=ops/t
	print("Numpy running time: %f  gflops: %f" % (t,gflops))

def gemm_gpu_v1(M,N,K,target):

    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")
	
	
    # Default schedule
    s = te.create_schedule(C.op)
	
    bx, by, tx, ty = s[C].tile(C.op.axis[0], C.op.axis[1], x_factor=32, y_factor=32)

    s[C].bind(bx, te.thread_axis("blockIdx.x"))
    s[C].bind(tx, te.thread_axis("threadIdx.x"))
    s[C].bind(by, te.thread_axis("blockIdx.y"))
    s[C].bind(ty, te.thread_axis("threadIdx.y"))
    

    func = tvm.build(s, [A, B, C], target=target, name="mmul_gpu")
    assert func
    print("\n------------CODIGO GENERADO------------\n")
    print(tvm.lower(s, [A, B, C], simple_mode=True))
    return func


def gemm_gpu_v2(M,N,K,B,target):

    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")
	
	
    # Default schedule
    s = te.create_schedule(C.op)
	
    bx, by, tx, ty = s[C].tile(C.op.axis[0], C.op.axis[1], x_factor=32, y_factor=32)

    s[C].bind(bx, te.thread_axis("blockIdx.y"))
    s[C].bind(tx, te.thread_axis("threadIdx.y"))
    s[C].bind(by, te.thread_axis("blockIdx.x"))
    s[C].bind(ty, te.thread_axis("threadIdx.x"))
    
    func = tvm.build(s, [A, B, C], target=target, name="mmul_gpu")
    assert func
    print("\n------------CODIGO GENERADO------------\n")
    print(tvm.lower(s, [A, B, C], simple_mode=True))
    return func


def gemm_gpu_v3(M,N,K,target):

    k = te.reduce_axis((0, K), "k")
    A = te.placeholder((M, K), name="A")
    B = te.placeholder((K, N), name="B")
    ths=32

    C = te.compute((M, N), lambda x, y: te.sum(A[x, k] * B[k, y], axis=k), name="C")
	
	
    # Default schedule
    s = te.create_schedule(C.op)
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis((0, ths), "threadIdx.x")
    thread_y = te.thread_axis((0, ths), "threadIdx.y")
    thread_xz = te.thread_axis((0, 2), "vthread", name="vx")
    thread_yz = te.thread_axis((0, 2), "vthread", name="vy")

    xx,yy = s[C].op.axis
   
    bx, tx = s[C].split(xx, factor=ths)
    by, ty = s[C].split(yy, factor=ths)
    
    s[C].bind(bx, block_y)
    s[C].bind(tx, thread_y)
    s[C].bind(by, block_x)
    s[C].bind(ty, thread_x)

    AA = s.cache_read(A, "shared", [C])
    s[AA].compute_at(s[C],tx)
    
    s[C].reorder(bx,by,k,tx,ty)


    func = tvm.build(s, [A, B, C], target=target, name="mmul_gpu")
    assert func
    print("\n------------CODIGO GENERADO------------\n")
    print(tvm.lower(s, [A, B, C], simple_mode=True))
    return func

def performance_gpu_tvm(ff, M,N,K,target_gpu,target_host,dtype,version):

    dev_h = tvm.device(target_host.kind.name, 0)
    a = tvm.nd.array(numpy.random.rand(M, K).astype(dtype), dev_h)
    b = tvm.nd.array(numpy.random.rand(K, N).astype(dtype), dev_h)
	
    dev_g = tvm.device(target_gpu.kind.name, 0)
    d_a = tvm.nd.array(a, dev_g)
    d_b = tvm.nd.array(b, dev_g)
    d_c = tvm.nd.array(numpy.zeros((M, N), dtype=dtype), dev_g)
    ff(d_a, d_b, d_c)

    answer = numpy.dot(a.numpy(), b.numpy())
    tvm.testing.assert_allclose(d_c.numpy(), answer, rtol=1e-5)
    evaluator = func.time_evaluator(func.entry_name, dev_g, number=100)
    mean_time = evaluator(d_a, d_b, d_c).mean
    ops=2.0*M*N*K/1e9

    print("%s GPU time: %f gflops: %f" % (version,mean_time, ops/mean_time))

if __name__ == "__main__":
    
    target_gpu = tvm.target.Target(target="cuda -arch=sm_72", host="llvm")
    target_host = tvm.target.Target(target="llvm", host="llvm")
    M = 4096
    K = 4096
    N = 4096
    M=N=K=2048 
    dtype = "float32"
    """
    performance_numpy(M,N,K)
    func = gemm_gpu_v1(M,N,K,target_gpu)
    performance_gpu_tvm(func,M,N,K,target_gpu,target_host,dtype,"V1")
    func = gemm_gpu_v2(M,N,K,target_gpu)
    performance_gpu_tvm(func,M,N,K,target_gpu,target_host,dtype,"V2")
    """
    func = gemm_gpu_v3(M,N,K,target_gpu)
    performance_gpu_tvm(func,M,N,K,target_gpu,target_host,dtype,"V3")
	
	
	


 

