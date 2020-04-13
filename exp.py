import cupy as cp



myk = r'''
extern "C"{

 __global__ void myrelu(float* x1,float* y, unsigned int N)
 {
     unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
     float tmp = x1[tid];
     if (tid < N  )
     {
   
        if (tmp<0){
         y[tid] = 0;}
     }
 }

 }'''

module = cp.cuda.compiler.compile_with_cache(myk)
myk = module.get_function('myrelu')
N = 5
x1 = cp.arange(-10,15,1, dtype=cp.float32).reshape(N,N)
x2 = cp.ones((N, N), dtype=cp.float32)
b = cp.ones((N, N), dtype=cp.float32)
myk((N,),(N,), (x1, b, N**2))
print(x1)
print(b)