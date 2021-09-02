
#include<iostream>
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<stdlib.h>
#include<stdio.h>
#include<thrust/sort.h>
#include<math.h>
#include<cuda.h>
using namespace std;
__global__ void k1(long *gdata,long *gquery,long *gres,int *gid,int N,int count) {
    int id = blockIdx.x*blockDim.x+threadIdx.x;;
    //gres[id*2+0] = id;
    gid[id] = id;
    float dist = 0;
    for(int i=1;i<count-1;i++){
        //printf("%d\t%0.2f\t%0.2f\n",id,gdata[id*count+i],gquery[i]);
        dist += (gdata[id*count+i]-gquery[i])*(gdata[id*count+i]-gquery[i]);
    }
    gres[id] = sqrt(dist);
    //printf("%d %0.2f\n",id,gres[id]);
}
__global__ void maxk(long *data,long *query,long *res,int *gid,int N,int count){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int i = id%N;
    int j = id/N;
    float dis = 0;
    for(int k=1;k<count-1;k++){
        dis +=((data[i*count+k]-query[j*count+k])*(data[i*count+k]-query[j*count+k]));
    }
    //printf("%d\n",id);
    res[id] = sqrt(dis);
    gid[id] = id;
}
__global__ void Accuracy(long *query,long *result,int count,int *counter){
    int id = threadIdx.x;
    //printf("%d %d\n",s1[id],s2[id]);
    int x = 1;
    if(query[id*count+10]==result[id]){
        atomicAdd(&counter[0],x);
    }
}

int main(){
    int k = 3 ;
    FILE *fp;
    int N = 10000;
    int count = 11;
    fp = fopen("binput.txt","r");
    char ch = ' ';
    long *data = (long *)malloc(N*count*sizeof(long));
    for(int i=0;i<N;i++){
        for(int j=0;j<count;j++){
            fscanf(fp,"%ld",&data[i*count+j]);
            ch = fgetc(fp);
            //cout<<data[i*count+j]<<"\t";
        }
        //cout<<"\n";
    }
    int m = 100;
    count = 11;
    FILE *op;
    op = fopen("bitest.txt","r");
    long *query = (long *)malloc(m*count*sizeof(long));
    for(int i=0;i<m;i++){
        for(int j=0;j<count;j++){
            fscanf(op,"%ld",&query[i*count+j]);
            ch = fgetc(op);
            //cout<<query[i*count+j]<<"\t";
        }
        //cout<<"\n";
    }
    long *result = (long *)malloc(m*sizeof(long));
    long *gquery,*gdata,*res,*gres;
    int *id,*gid;
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    float ms = 0;
    for(int i=0;i<m;i++){
        long *point = (long *)malloc(count*sizeof(long));
        for(int j=0;j<count;j++){
            point[j] = query[i*count+j];
        }
        float milliseconds1 = 0;
        cudaEventRecord(start1,0);
        cudaMalloc(&gquery,count*sizeof(long));
        cudaMalloc(&gdata,N*count*sizeof(long));
        cudaMalloc(&gres,N*sizeof(long));
        cudaMalloc(&gid,N*sizeof(int));
        res = (long *)malloc(N*sizeof(long));
        id = (int *)malloc(N*sizeof(int));
        cudaMemcpy(gdata,data,N*count*sizeof(long),cudaMemcpyHostToDevice);
        cudaMemcpy(gquery,point,count*sizeof(long),cudaMemcpyHostToDevice);

        //Launching one test point to all train point kernal


        k1<<<16,N/16>>>(gdata,gquery,gres,gid,N,count);
        cudaMemcpy(res,gres,N*sizeof(long),cudaMemcpyDeviceToHost);
        cudaMemcpy(id,gid,N*sizeof(int),cudaMemcpyDeviceToHost);
        cudaEventRecord(stop1,0);
        cudaEventSynchronize(stop1);
        cudaEventElapsedTime(&milliseconds1, start1, stop1);
        ms+=milliseconds1;
        thrust::sort_by_key(res, res + N, id);
        //cout<<"\n============================\n";
        int count1,count2;
        count1 = count2 = 0;
        for(int j=0;j<k;j++){
            //cout<<i<<" "<<id[j]<<" "<<res[j]<<"\n";
            //cout<<id[j]<<" "<<data[id[j]*count+10]<<"\n";
            if(data[id[j]*count+10]==2){
                count1++;
            }
            if(data[id[j]*count+10]==4){
                count2++;
            }
            
        }
        //cout<<count1<<" "<<count2<<"\n";
        if(count1>count2){
            result[i] = 2;
        }
        else{
            result[i] = 4;
        }
    }
    int *gcounter;
    int counter[1];
    long *gresult,*ggquery;
    cudaMalloc(&gresult,m*sizeof(long));
    cudaMalloc(&ggquery,m*count*sizeof(long));
    counter[0] = 0;
    cudaMalloc(&gcounter,1*sizeof(int));

    cudaMemcpy(gcounter,counter,1*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(ggquery,query,m*count*sizeof(long),cudaMemcpyHostToDevice);
    cudaMemcpy(gresult,result,m*sizeof(long),cudaMemcpyHostToDevice);

    Accuracy<<<1,m>>>(ggquery,gresult,count,gcounter);
    cudaMemcpy(counter,gcounter,1*sizeof(int),cudaMemcpyDeviceToHost);
    
    printf(" Total time taken %f\n",ms);
    //cout<<counter[0];
    float acc = counter[0]*100;
    acc = acc/m;
    cout<<"Accuracy of KNN "<<acc<<"\n";
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    cudaEventRecord(start,0);

    
    int *id2d,*gid2d;
    long *gdata2d,*gquery2d,*gres2d,*res2d;
    cudaMalloc(&gid2d,m*N*sizeof(int));
    id2d = (int *)malloc(m*N*sizeof(int));
    res2d = (long *)malloc(m*N*sizeof(long));
    cudaMalloc(&gres2d,m*N*sizeof(long));
    cudaMalloc(&gdata2d,N*count*sizeof(long));
    cudaMalloc(&gquery2d,m*count*sizeof(long));
    cudaMemcpy(gdata2d,data,N*count*sizeof(long),cudaMemcpyHostToDevice);
    cudaMemcpy(gquery2d,query,m*count*sizeof(long),cudaMemcpyHostToDevice);
    
    maxk<<<16*m,N/16>>>(gdata2d,gquery2d,gres2d,gid2d,N,count);
    
    cudaMemcpy(id2d,gid2d,m*N*sizeof(int),cudaMemcpyDeviceToHost);
    cudaMemcpy(res2d,gres2d,m*N*sizeof(long),cudaMemcpyDeviceToHost);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Total time taken %f\n",milliseconds);
    for(int i=0;i<m;i++){
        //cout<<"Line"<<i<<"\t";
        long *dist = (long *)malloc(N*sizeof(long));
        int *im = (int *)malloc(N*sizeof(int));
        for(int j=0;j<N;j++){
            //cout<<res2d[i*N+j]<<"\t";
            im[j] = id2d[i*N+j]%N;
            dist[j] = res2d[i*N+j];
        }
        thrust::sort_by_key(dist, dist + N, im);
        int count1,count2;
        count1 = count2 = 0;
        for(int j=0;j<k;j++){
            //cout<<im[j]<<"\t";
            if(data[im[j]*count+10]==2){
                count1++;
            }
            if(data[im[j]*count+10]==4){
                count2++;
            }
        }
        if(count1>count2){
            result[i] = 2;
        }
        else{
            result[i] = 4;
        }
        //cout<<result[i]<<"\n";
        //cout<<count1<<" "<<count2<<"\n";
    }

    
    int *ggcounter;
    int ccounter[1];
    long *ggresult,*gggquery;
    cudaMalloc(&ggresult,m*sizeof(long));
    cudaMalloc(&gggquery,m*count*sizeof(long));
    ccounter[0] = 0;
    cudaMalloc(&ggcounter,1*sizeof(int));

    cudaMemcpy(ggcounter,ccounter,1*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gggquery,query,m*count*sizeof(long),cudaMemcpyHostToDevice);
    cudaMemcpy(ggresult,result,m*sizeof(long),cudaMemcpyHostToDevice);

    Accuracy<<<1,m>>>(gggquery,ggresult,count,ggcounter);
    cudaMemcpy(ccounter,ggcounter,1*sizeof(int),cudaMemcpyDeviceToHost);
    
    float acc1 = ccounter[0]*100;
    acc1 = acc1/m;

    cout<<"Accuracy of KNN "<<acc1<<"\n";

    cudaDeviceSynchronize();
    return 0;
}