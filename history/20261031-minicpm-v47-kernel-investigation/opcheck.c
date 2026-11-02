#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include "ggml.h"
#ifdef UPSTREAM
#include "gguf.h"
#include "ggml-cpu.h"
#endif
extern void dequantize_row_q4_K(const void *,float *,int64_t);
int main(int argc,char**argv){
 if(argc!=2)return 2;struct gguf_init_params gp={.no_alloc=true,.ctx=NULL};struct gguf_context*g=gguf_init_from_file(argv[1],gp);if(!g)return 3;int ix=gguf_find_tensor(g,"blk.0.ffn_gate_exps.weight");if(ix<0)return 4;size_t off=gguf_get_data_offset(g)+gguf_get_tensor_offset(g,ix);gguf_free(g);
 const int K=2048,M=512;struct ggml_init_params ip={.mem_size=64*1024*1024,.mem_buffer=NULL,.no_alloc=false};struct ggml_context*ctx=ggml_init(ip);if(!ctx)return 5;struct ggml_tensor*w=ggml_new_tensor_2d(ctx,GGML_TYPE_Q4_K,K,M);struct ggml_tensor*x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,K,1);struct ggml_tensor*y=ggml_mul_mat(ctx,w,x);struct ggml_cgraph*graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,y);FILE*fp=fopen(argv[1],"rb");if(!fp)return 6;if(fseeko(fp,off,SEEK_SET))return 7;size_t n=ggml_nbytes(w);if(fread(w->data,1,n,fp)!=n)return 8;fclose(fp);float*deq=malloc(K*sizeof(float));if(!deq)return 9;
 for(int k=0;k<3;k++){float*in=x->data;for(int i=0;i<K;i++){float v=sinf((float)(i+1)*(0.031f+0.019f*k))+0.25f*cosf((float)i*0.13f);if(k==1 && i%97==0)v*=12; if(k==2 && i%113==0)v*=40;in[i]=v;}if(ggml_graph_compute_with_ctx(ctx,graph,16)!=GGML_STATUS_SUCCESS)return 10;double se=0,sr=0,ma=0;float*out=y->data;for(int j=0;j<M;j++){dequantize_row_q4_K((const char*)w->data+j*w->nb[1],deq,K);double ref=0;for(int i=0;i<K;i++)ref+=(double)deq[i]*in[i];double er=out[j]-ref;se+=er*er;sr+=ref*ref;if(fabs(er)>ma)ma=fabs(er);}printf("pattern=%d nrmse=%.9g maxabs=%.9g\n",k,sqrt(se/sr),ma);}
 free(deq);ggml_free(ctx);return 0;
}
