// Explicit inference primitives. Every launch uses the session's one stream.
// No Torch, serialized program, runtime compiler, or global model registry.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <new>
#include <cstdint>
#include <cstdlib>

static thread_local char last_error[512] = {};
#define CUDA(call) do { auto e = (call); if (e != cudaSuccess) { \
    snprintf(last_error, sizeof(last_error), "%s: %s", #call, cudaGetErrorString(e)); return -1; } } while (0)
#define BLAS(call) do { auto e = (call); if (e != CUBLAS_STATUS_SUCCESS) { \
    snprintf(last_error, sizeof(last_error), "%s: cuBLAS status %d", #call, (int)e); return -1; } } while (0)

struct Session { cudaStream_t stream{}; cublasHandle_t blas{}; };
extern "C" const char* glados_error() { return last_error; }
extern "C" int glados_create(Session** out) {
    *out = nullptr;
    CUDA(cudaSetDevice(0));
    cudaMemPool_t pool;
    CUDA(cudaDeviceGetDefaultMemPool(&pool,0));
    // Bound the process-local retained pool. A zero override reproduces the
    // original release-on-synchronization behavior for paired measurements.
    uint64_t retained=512ull*1024*1024;
    if(const char* setting=std::getenv("GLADOS_CUDA_POOL_LIMIT_MB")) {
        char* end=nullptr;auto mb=std::strtoull(setting,&end,10);
        if(!*setting || *end || mb>4096) {snprintf(last_error,sizeof(last_error),"GLADOS_CUDA_POOL_LIMIT_MB must be in 0..4096");return -1;}
        retained=mb*1024*1024;
    }
    CUDA(cudaMemPoolSetAttribute(pool,cudaMemPoolAttrReleaseThreshold,&retained));
    Session* s = new(std::nothrow) Session;
    if (!s) { snprintf(last_error, sizeof(last_error), "session allocation failed"); return -1; }
    auto e = cudaStreamCreateWithFlags(&s->stream, cudaStreamNonBlocking);
    if (e != cudaSuccess) { delete s; snprintf(last_error, sizeof(last_error), "%s", cudaGetErrorString(e)); return -1; }
    auto b = cublasCreate(&s->blas);
    if (b != CUBLAS_STATUS_SUCCESS) { cudaStreamDestroy(s->stream); delete s; snprintf(last_error, sizeof(last_error), "cublasCreate: %d", (int)b); return -1; }
    b = cublasSetStream(s->blas, s->stream);
    if (b != CUBLAS_STATUS_SUCCESS) { cublasDestroy(s->blas); cudaStreamDestroy(s->stream); delete s; return -1; }
    *out = s;
    return 0;
}
extern "C" void glados_destroy(Session* s) {
    if (s) { cudaStreamSynchronize(s->stream); cublasDestroy(s->blas); cudaStreamDestroy(s->stream); delete s; }
}
extern "C" int glados_alloc(Session* s, size_t bytes, void** p) {
    CUDA(cudaMallocAsync(p, bytes, s->stream)); return 0;
}
extern "C" void glados_free(Session* s, void* p) { if (p) cudaFreeAsync(p, s->stream); }
extern "C" int glados_upload(Session* s, void* dst, const void* src, size_t bytes) {
    CUDA(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, s->stream));
    // Host slice may cease to exist when this FFI call returns.
    CUDA(cudaStreamSynchronize(s->stream)); return 0;
}
extern "C" int glados_download(Session* s, void* dst, const void* src, size_t bytes) {
    CUDA(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, s->stream));
    CUDA(cudaStreamSynchronize(s->stream)); return 0;
}
extern "C" int glados_sync(Session* s) { CUDA(cudaStreamSynchronize(s->stream)); return 0; }
extern "C" void* glados_stream(Session* s) { return s->stream; }
extern "C" int glados_zero(Session* s,void* x,size_t bytes) {
    CUDA(cudaMemsetAsync(x,0,bytes,s->stream)); return 0;
}

__device__ float act(float x, float slope) { return x >= 0 ? x : slope*x; }
// Weight layout [out, in, tap], activation layout [channel, time].
__global__ void gather_conv(const float* x, float* col, int channels, int time,
                            int kernel, int dilation, int padding, float slope) {
    size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    size_t count = (size_t)channels*kernel*time;
    if (i >= count) return;
    int t = i % time, row = i / time, k = row % kernel, c = row / kernel;
    int source = t + k*dilation - padding;
    col[i] = source >= 0 && source < time ? act(x[(size_t)c*time+source], slope) : 0.f;
}
__global__ void finish_conv(float* y, const float* bias, const float* residual, int time, size_t count) {
    size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (i < count) y[i] += bias[i/time] + (residual ? residual[i] : 0.f);
}
extern "C" int glados_conv(Session* s, const float* x, const float* w, const float* bias,
                            const float* residual, float* col, float* y,
                            int in, int out, int time, int kernel, int dilation, int padding, float slope) {
    size_t count = (size_t)in*kernel*time;
    const float* matrix=x;
    if(kernel!=1 || slope!=1.f) {
        gather_conv<<<(count+255)/256, 256, 0, s->stream>>>(x,col,in,time,kernel,dilation,padding,slope);
        CUDA(cudaGetLastError());matrix=col;
    }
    float alpha=1.f, beta=0.f;
    BLAS(cublasSgemm(s->blas,CUBLAS_OP_N,CUBLAS_OP_N,time,out,in*kernel,
                     &alpha,matrix,time,w,in*kernel,&beta,y,time));
    count = (size_t)out*time;
    finish_conv<<<(count+255)/256,256,0,s->stream>>>(y,bias,residual,time,count);
    CUDA(cudaGetLastError()); return 0;
}

// Polyphase transpose convolution: two input taps per output phase; avoid
// materializing stride-1 zeros or doing a GEMM over the expanded time axis.
__global__ void gather_up(const float* x, float* col, int channels, int time, int stride, int padding) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    size_t phase_size=(size_t)2*channels*time;
    if(i>=phase_size*stride) return;
    int phase=i/phase_size, t=i%time, row=(i%phase_size)/time;
    int c=row/2, tap=row%2;
    int k=(phase+padding)%stride + tap*stride;
    int source=t+(phase+padding-k)/stride;
    col[i]=source>=0 && source<time ? act(x[(size_t)c*time+source],0.1f) : 0.f;
}
__global__ void finish_up(const float* phases, const float* bias, float* y, int out, int time, int stride) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    int result_time=time*stride;
    if(i >= (size_t)out*result_time) return;
    int c=i/result_time, t=i%result_time, phase=t%stride;
    y[i]=phases[((size_t)phase*out+c)*time+t/stride]+bias[c];
}
extern "C" int glados_up(Session* s, const float* x, const float* w, const float* bias,
                         float* col, float* phases, float* y, int in, int out, int time, int stride) {
    size_t count=(size_t)stride*2*in*time;
    gather_up<<<(count+255)/256,256,0,s->stream>>>(x,col,in,time,stride,stride/2);
    CUDA(cudaGetLastError());
    float alpha=1.f,beta=0.f;
    BLAS(cublasSgemmStridedBatched(s->blas,CUBLAS_OP_N,CUBLAS_OP_N,time,out,2*in,
        &alpha,col,time,(long long)2*in*time,w,2*in,(long long)out*2*in,
        &beta,phases,time,(long long)out*time,stride));
    count=(size_t)out*time*stride;
    finish_up<<<(count+255)/256,256,0,s->stream>>>(phases,bias,y,out,time,stride);
    CUDA(cudaGetLastError()); return 0;
}
__global__ void mean3(const float* a,const float* b,const float* c,float* out,size_t n) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) out[i]=((a[i]+b[i])+c[i])/3.f;
}
extern "C" int glados_mean3(Session* s,const float* a,const float* b,const float* c,float* out,size_t n) {
    mean3<<<(n+255)/256,256,0,s->stream>>>(a,b,c,out,n); CUDA(cudaGetLastError()); return 0;
}
__global__ void tanh_output(float* x,size_t n) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) x[i]=tanhf(x[i]);
}
extern "C" int glados_tanh(Session* s,float* x,size_t n) {
    tanh_output<<<(n+255)/256,256,0,s->stream>>>(x,n); CUDA(cudaGetLastError()); return 0;
}

extern "C" int glados_copy(Session* s,void* dst,const void* src,size_t bytes) {
    CUDA(cudaMemcpyAsync(dst,src,bytes,cudaMemcpyDeviceToDevice,s->stream)); return 0;
}
__global__ void transpose_matrix(const float* x,float* y,int rows,int cols) {
    __shared__ float tile[32][33];
    int c=blockIdx.x*32+threadIdx.x,r=blockIdx.y*32+threadIdx.y;
    for(int j=0;j<32;j+=8) if(c<cols && r+j<rows) tile[threadIdx.y+j][threadIdx.x]=x[(size_t)(r+j)*cols+c];
    __syncthreads();
    c=blockIdx.y*32+threadIdx.x; r=blockIdx.x*32+threadIdx.y;
    for(int j=0;j<32;j+=8) if(c<rows && r+j<cols) y[(size_t)(r+j)*rows+c]=tile[threadIdx.x][threadIdx.y+j];
}
extern "C" int glados_transpose(Session* s,const float* x,float* y,int rows,int cols) {
    transpose_matrix<<<dim3((cols+31)/32,(rows+31)/32),dim3(32,8),0,s->stream>>>(x,y,rows,cols);
    CUDA(cudaGetLastError()); return 0;
}
__global__ void embedding(const float* w,const int* ids,float* y,int time,int width) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<(size_t)time*width) y[i]=w[(size_t)ids[i%time]*width+i/time];
}
extern "C" int glados_embedding(Session* s,const float* w,const int* ids,float* y,int time,int width) {
    size_t n=(size_t)time*width; embedding<<<(n+255)/256,256,0,s->stream>>>(w,ids,y,time,width);
    CUDA(cudaGetLastError()); return 0;
}
__global__ void broadcast_vector(const float* x,float* y,int time,int channels) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<(size_t)time*channels) y[i]=x[i/time];
}
extern "C" int glados_broadcast(Session* s,const float* x,float* y,int time,int channels) {
    size_t n=(size_t)time*channels; broadcast_vector<<<(n+255)/256,256,0,s->stream>>>(x,y,time,channels);
    CUDA(cudaGetLastError()); return 0;
}
__global__ void batch_norm(float* x,const float* scale,const float* bias,int time,size_t n,int relu) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) { float value=x[i]; if(relu) value=fmaxf(value,0.f); x[i]=value*scale[i/time]+bias[i/time]; }
}
extern "C" int glados_bn(Session* s,float* x,const float* scale,const float* bias,int time,size_t n,int relu) {
    batch_norm<<<(n+255)/256,256,0,s->stream>>>(x,scale,bias,time,n,relu);
    CUDA(cudaGetLastError()); return 0;
}
__global__ void max_pool(const float* x,float* y,int time,size_t n) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) y[i]=i%time==0 ? x[i] : fmaxf(x[i-1],x[i]);
}
extern "C" int glados_pool(Session* s,const float* x,float* y,int time,size_t n) {
    max_pool<<<(n+255)/256,256,0,s->stream>>>(x,y,time,n); CUDA(cudaGetLastError()); return 0;
}
__global__ void highway(const float* x,const float* a,const float* b,float* y,size_t n) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) {float g=1.f/(1.f+expf(-b[i]));y[i]=g*fmaxf(a[i],0.f)+(1.f-g)*x[i];}
}
extern "C" int glados_highway(Session* s,const float* x,const float* a,const float* b,float* y,size_t n) {
    highway<<<(n+255)/256,256,0,s->stream>>>(x,a,b,y,n); CUDA(cudaGetLastError()); return 0;
}
__global__ void add_values(const float* a,const float* b,float* y,size_t n) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]=a[i]+b[i];
}
extern "C" int glados_add(Session* s,const float* a,const float* b,float* y,size_t n) {
    add_values<<<(n+255)/256,256,0,s->stream>>>(a,b,y,n); CUDA(cudaGetLastError()); return 0;
}
__global__ void argmax_channels(const float* x,int* y,int time,int channels) {
    int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t<time) {int best=0;float value=x[t];for(int c=1;c<channels;c++) if(x[(size_t)c*time+t]>value){best=c;value=x[(size_t)c*time+t];}y[t]=best;}
}
extern "C" int glados_argmax(Session* s,const float* x,int* y,int time,int channels) {
    argmax_channels<<<(time+255)/256,256,0,s->stream>>>(x,y,time,channels); CUDA(cudaGetLastError()); return 0;
}
__global__ void gather_columns(const float* x,const int* ids,float* y,int input_time,int output_time,int channels) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<(size_t)channels*output_time) y[i]=x[(i/output_time)*input_time+ids[i%output_time]];
}
extern "C" int glados_gather(Session* s,const float* x,const int* ids,float* y,int input_time,int output_time,int channels) {
    size_t n=(size_t)channels*output_time;gather_columns<<<(n+255)/256,256,0,s->stream>>>(x,ids,y,input_time,output_time,channels);
    CUDA(cudaGetLastError()); return 0;
}

__global__ void phoneme_embed(const float* w,const int* ids,const float* pos,float scale,float* y,int time) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<(size_t)time*512) {int t=i%time,c=i/time;y[i]=w[(size_t)ids[t]*512+c]+pos[(size_t)t*512+c]*scale;}
}
extern "C" int glados_phoneme_embed(Session* s,const float* w,const int* ids,const float* pos,float scale,float* y,int time) {
    size_t n=(size_t)time*512;phoneme_embed<<<(n+255)/256,256,0,s->stream>>>(w,ids,pos,scale,y,time);
    CUDA(cudaGetLastError());return 0;
}
__device__ float reduce_sum(float x,float* shared) {
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    for(int offset=16;offset;offset/=2)x+=__shfl_down_sync(0xffffffff,x,offset);
    if(lane==0)shared[warp]=x;
    __syncthreads();
    x=threadIdx.x<blockDim.x/32?shared[lane]:0.f;
    if(warp==0)for(int offset=16;offset;offset/=2)x+=__shfl_down_sync(0xffffffff,x,offset);
    if(threadIdx.x==0)shared[0]=x;
    __syncthreads();return shared[0];
}
__device__ float reduce_max(float x,float* shared) {
    int lane=threadIdx.x%32,warp=threadIdx.x/32;
    for(int offset=16;offset;offset/=2)x=fmaxf(x,__shfl_down_sync(0xffffffff,x,offset));
    if(lane==0)shared[warp]=x;
    __syncthreads();
    x=threadIdx.x<blockDim.x/32?shared[lane]:-INFINITY;
    if(warp==0)for(int offset=16;offset;offset/=2)x=fmaxf(x,__shfl_down_sync(0xffffffff,x,offset));
    if(threadIdx.x==0)shared[0]=x;
    __syncthreads();return shared[0];
}
__global__ void softmax_rows(float* scores,int time) {
    __shared__ float shared[8];
    float* row=scores+(size_t)blockIdx.x*time;
    float maxval=-INFINITY;
    for(int i=threadIdx.x;i<time;i+=blockDim.x)maxval=fmaxf(maxval,row[i]);
    maxval=reduce_max(maxval,shared);
    float sum=0;
    for(int i=threadIdx.x;i<time;i+=blockDim.x){float e=expf(row[i]-maxval);row[i]=e;sum+=e;}
    // All threads have consumed the maximum before shared storage is reused.
    __syncthreads();sum=reduce_sum(sum,shared);
    for(int i=threadIdx.x;i<time;i+=blockDim.x)row[i]/=sum;
}
extern "C" int glados_attention(Session* s,const float* q,const float* k,const float* v,float* scores,float* y,int time) {
    float alpha=1.f/11.313708f,beta=0.f;
    BLAS(cublasSgemmStridedBatched(s->blas,CUBLAS_OP_N,CUBLAS_OP_T,time,time,128,
        &alpha,k,time,(long long)128*time,q,time,(long long)128*time,&beta,scores,time,(long long)time*time,4));
    softmax_rows<<<4*time,256,0,s->stream>>>(scores,time);CUDA(cudaGetLastError());
    alpha=1.f;
    BLAS(cublasSgemmStridedBatched(s->blas,CUBLAS_OP_T,CUBLAS_OP_N,time,128,time,
        &alpha,scores,time,(long long)time*time,v,time,(long long)128*time,&beta,y,time,(long long)128*time,4));
    return 0;
}
__global__ void layer_norm(const float* x,const float* residual,const float* gamma,const float* beta,float* y,int time) {
    __shared__ float shared[16];
    int t=blockIdx.x,c=threadIdx.x;
    size_t i=(size_t)c*time+t;
    float value=x[i]+(residual?residual[i]:0.f);
    float mean=reduce_sum(value,shared)/512.f;
    float delta=value-mean;
    __syncthreads();float variance=reduce_sum(delta*delta,shared)/512.f;
    y[i]=(delta*rsqrtf(variance+1e-5f))*gamma[c]+beta[c];
}
extern "C" int glados_layernorm(Session* s,const float* x,const float* residual,const float* gamma,const float* beta,float* y,int time) {
    layer_norm<<<time,512,0,s->stream>>>(x,residual,gamma,beta,y,time);CUDA(cudaGetLastError());return 0;
}
