
//-------------------------------------GPU Implementation of KNN--------------------------------------------------
//---------------------------Train Data store in input.txt and Test data in test.txt------------------------------

#include<iostream>
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<stdlib.h>
#include<stdio.h>
#include<thrust/sort.h>
#include<math.h>
#include<cuda.h>
using namespace std;

// Calculating distance in parallel for one test point and all training point 
// Kernal launched with 1*n threads

__global__ void k1(float *gdata,float *gquery,float *gres,int *gid,int N,int count) {
    int id = threadIdx.x;
    //gres[id*2+0] = id;
    gid[id] = id;
    float dist = 0;
    for(int i=1;i<count;i++){
        //printf("%d\t%0.2f\t%0.2f\n",id,gdata[id*count+i],gquery[i]);
        dist += (gdata[id*count+i]-gquery[i])*(gdata[id*count+i]-gquery[i]);
    }
    gres[id] = sqrt(dist);
    //printf("%d %0.2f\n",id,gres[id]);
}
/*__global__ void k(float *data,int N,int count){
        for(int j=0;j<count;j++){
            printf("%d\n",data[threadIdx.x*count+j]);
        }
}*/

//Calculating distances in parallel between all train point and test point .
//kernal launched with m*n threads


__global__ void maxkernal(float *data,float *query,float *dis,int *gid,int N,int count){
    int id = blockIdx.x*blockDim.x+threadIdx.x;
    int i = id/N;
    int j = id%N;
    //float diss = 0;
    for(int k=1;k<count;k++){
        //printf("%d %0.2f %0.2f %0.2f %0.2f\n",id,data[j*count+k],query[i*count+k],(data[j*count+k]-query[i*count+k]),dis[id]);
        atomicAdd(&dis[id],((data[j*count+k]-query[i*count+k])*(data[j*count+k]-query[i*count+k])));
        //printf("%d %0.2f %0.2f %0.2f %0.2f %0.2f\n",id,data[j*count+k],query[i*count+k],(data[j*count+k]-query[i*count+k]),dis[id],((data[j*count+k]-query[i*count+k])*(data[j*count+k]-query[i*count+k])));
    }
    gid[id] = id;
    dis[id] = sqrt(dis[id]);
}

// Accuracy calculation in parallel

__global__ void Accuracy(int *s1,int *s2,int *counter){
    int id = threadIdx.x;
    //printf("%d %d\n",s1[id],s2[id]);
    int x = 1;
    if(s1[id]==s2[id]){
        atomicAdd(&counter[0],x);
    }
}

// Begin of the main function 


int main(){


    //Reading the train points


    int k=15;
    int N=135;
    int count=0;
    FILE *fp;
    string s[N];
    fp = fopen("input.txt","r");
    char ch = ' ';
    while(ch!='\n'){
        ch = getc(fp);
        if(ch==','){
        count++;
        }
    }
    float *data = (float *)malloc(N*count*sizeof(float));
    for(int i=0;i<N;i++){
        for(int j=0;j<count;j++){
            fscanf(fp,"%f",&data[i*count+j]);
            ch = fgetc(fp);
            //cout<<data[i*count+j]<<"\t";
        }
        char c;
        c = fgetc(fp);
        while(c!='\n'){
            s[i] += c;
            c = fgetc(fp);
        }
        //cout<<s[i]<<"\n";
    }
    fclose(fp);
    float *gdata,*gres,*res;
    int *id,*gid;
    int *fclass;
    /*cudaMalloc(&gdata,N*count*sizeof(float));
    cudaMemcpy(gdata,data,N*count*sizeof(float),cudaMemcpyHostToDevice);
    k<<<1,N>>>(gdata,N,count);*/
    //cout<<"----------------------------------------------------\n";
    
    
    //Reading the test point 


    FILE *op;
    int m=15;
    string s1[m];
    int gsres[m];
    float *query,*gquery;
    float *query2d = (float *)malloc(m*count*sizeof(float));
    fclass = (int *)malloc(m*sizeof(int));
    op = fopen("test.txt","r");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float ms = 0;

    for(int i=0;i<m;i++){
        query = (float *)malloc(count*sizeof(float));
        for(int j=0;j<count;j++){
            fscanf(op,"%f",&query[j]);
            query2d[i*count+j] = query[j];
            ch = fgetc(op);
            //cout<<query[i*count+j]<<"\t";
        }
        char c;
        c = fgetc(op);
        while(c!='\n'){
            s1[i] += c;
            c = fgetc(op);
        }
        if(s1[i]=="Iris-setosa"){
            fclass[i] = 1;
            //cout<<"c1";
        }
        if(s1[i]=="Iris-versicolor"){
            fclass[i] = 2;
            //cout<<"c2";
        }
        if(s1[i]=="Iris-virginica"){
            fclass[i] = 3;
            //cout<<"c3";
        }
        //cout<<s1[i]<<"\n";
        float milliseconds = 0;
        cudaEventRecord(start,0);
        cudaMalloc(&gquery,count*sizeof(float));
        cudaMalloc(&gdata,N*count*sizeof(float));
        cudaMalloc(&gres,N*sizeof(float));
        cudaMalloc(&gid,N*sizeof(int));
        res = (float *)malloc(N*sizeof(float));
        id = (int *)malloc(N*sizeof(int));
        cudaMemcpy(gdata,data,N*count*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(gquery,query,count*sizeof(float),cudaMemcpyHostToDevice);

        //Launching one test point to all train point kernal
        
        k1<<<1,N>>>(gdata,gquery,gres,gid,N,count);
        
        cudaMemcpy(res,gres,N*sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(id,gid,N*sizeof(int),cudaMemcpyDeviceToHost);
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        ms += milliseconds;
        thrust::sort_by_key(res, res + N, id);
        int count1,count2,count3;
        count1 = count2 = count3 = 0;

        //voting process of K closest neighbour


        for(int j=0;j<k;j++){
            //cout<<i<<" "<<minKarr[j][0]<<" "<<minKarr[j][1]<<"\n";
            if(s[id[j]]=="Iris-setosa"){
                count1++;
            }
            if(s[id[j]]=="Iris-versicolor"){
                count2++;
            }
            if(s[id[j]]=="Iris-virginica"){
                count3++;
            }
        }
        //cout<<count1<<" "<<count2<<" "<<count3<<"\n";
        if(count1>count2){
            if(count1>count3){
                //count1
                gsres[i] = 1;
            }
            else{
               //count3
                gsres[i] = 3;
            }
        }
        else{
           if(count2>count3){
              //count2
              gsres[i] = 2;
           }
           else{
               //count3
               gsres[i] = 3;
           }
        }
        //cout<<gsres[i]<<"\n";
        //cout<<"---------------------------------------------\n";
    }
    /*for(int i=0;i<m;i++){
        printf("%d\n",fclass[i]);
    }*/
    int *gclass,*ggsres,*gcounter;
    int counter[1];
    counter[0] = 0;
    cudaMalloc(&gclass,m*sizeof(int));
    cudaMalloc(&ggsres,m*sizeof(int));
    cudaMalloc(&gcounter,1*sizeof(int));
    cudaMemcpy(gclass,fclass,m*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(ggsres,gsres,m*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gcounter,counter,1*sizeof(int),cudaMemcpyHostToDevice);

    // Accuracy calculation 


    Accuracy<<<1,m>>>(gclass,ggsres,gcounter);
    cudaMemcpy(counter,gcounter,1*sizeof(int),cudaMemcpyDeviceToHost);
    //printf("%d\n",counter[0]);
    float acc = counter[0]*100;
    acc = acc/m;
    
    printf("Basic KNN Time taken in %f millisecond\n",ms);


    //cout<<"Time taken "<<elapsetime<<"\n";
    
    cout<<"Accuracy of KNN "<<acc<<"%"<<"\n";
    
    
    // prediction on random points
    srand(time(0));
    float *points = (float *)malloc(count*sizeof(float));
    for(int j=0;j<count;j++){
        if(j<count-1){
            points[j] = rand()%8;
        }
        else{
            points[j] = rand()%3;
        }
    }
    /*for(int j=0;j<count;j++){
        cout<<points[j]<<"\t";
    }*/
    cout<<"\n";    
    float *dis,*ggdata;
    float *gpoint,*gdis;
    int *gidd;
    int *idd;
    cudaMalloc(&gpoint,count*sizeof(float));
    cudaMalloc(&ggdata,N*count*sizeof(float));
    cudaMalloc(&gdis,N*sizeof(float));
    cudaMalloc(&gidd,N*sizeof(int));
    dis = (float *)malloc(N*sizeof(float));
    idd = (int *)malloc(N*sizeof(int));
    cudaMemcpy(ggdata,data,N*count*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(gpoint,points,count*sizeof(float),cudaMemcpyHostToDevice);

    //Launching one test point to all train point kernal

    k1<<<1,N>>>(gdata,gpoint,gdis,gidd,N,count);
    cudaMemcpy(dis,gdis,N*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(idd,gidd,N*sizeof(int),cudaMemcpyDeviceToHost);
    thrust::sort_by_key(dis, dis + N, idd);
    int count1,count2,count3; 
    count1 = count2 = count3 = 0;
    
    //voting process of K closest neighbour

    for(int i=0;i<k;i++){
        if(s[idd[i]]=="Iris-setosa"){
            count1++;
        }
        if(s[idd[i]]=="Iris-versicolor"){
            count2++;
        }
        if(s[idd[i]]=="Iris-virginica"){
            count3++;
        }
    }

    //Deciding on voting result

    string prediction;
    if(count1>count2){
        if(count1>count3){
            //count1
            prediction = "Iris-setosa";
        }
        else{
           //count3
           prediction = "Iris-virginica";
        }
   }
   else{
       if(count2>count3){
          //count2
          prediction = "Iris-versicolor";
       }
       else{
           //count3
           prediction = "Iris-virginica";
       }
   }
   cout<<"prediction Result "<<prediction<<"\n";
   

    // More parallelism 
    
    /*for(int i=0;i<m;i++){
        for(int j=0;j<count;j++){
            cout<<query2d[i*count+j]<<"\t";
        }
        cout<<"\n";
    }*/
    
    //One more Knn implementation
    cudaEvent_t start1, stop1;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    float milliseconds1 = 0;
    cudaEventRecord(start1,0);
    int *id2d,*gid2d;
    int *mres = (int *)malloc(m*sizeof(int));
    float *gquery2d,*gdatam,*gdist,*dist;
    cudaMalloc(&gquery2d,m*count*sizeof(float));
    cudaMemcpy(gquery2d,query2d,m*count*sizeof(float),cudaMemcpyHostToDevice);
    cudaMalloc(&gdatam,N*count*sizeof(float));
    cudaMemcpy(gdatam,data,N*count*sizeof(float),cudaMemcpyHostToDevice);
    dist = (float *)malloc(m*N*sizeof(float));
    cudaMalloc(&gdist,m*N*sizeof(float));
    id2d = (int *)malloc(m*N*sizeof(int));
    cudaMalloc(&gid2d,m*N*sizeof(int));

    //Distance calculation of KNN through all train and all test points in parallel
    //launching M*N threads
    

    maxkernal<<<m,N>>>(gdatam,gquery2d,gdist,gid2d,N,count);
    cudaMemcpy(dist,gdist,m*N*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(id2d,gid2d,m*N*sizeof(int),cudaMemcpyDeviceToHost);
    cudaEventRecord(stop1,0);
    cudaEventSynchronize(stop1);
    cudaEventElapsedTime(&milliseconds1, start1, stop1);
    
    for(int i=0;i<m;i++){
        float *distance = (float *)malloc(N*sizeof(float));
        int *index = (int *)malloc(N*sizeof(int));
        for(int j=0;j<N;j++){
            distance[j] = dist[i*N+j];
            index[j] = id2d[i*N+j];
        }

        //Sorting the K nearest neighbour.

        thrust::sort_by_key(distance, distance + N, index);
        int count1,count2,count3;
        
        //voting for K nearest neighbour

        count1 = count2 = count3 = 0;
        for(int j=0;j<k;j++){
            int p = index[j]%N;
                //cout<<i<<" "<<minKarr[j][0]<<" "<<minKarr[j][1]<<"\n";
                if(s[p]=="Iris-setosa"){
                    count1++;
                }
                if(s[p]=="Iris-versicolor"){
                    count2++;
                }
                if(s[p]=="Iris-virginica"){
                    count3++;
                }
        }
        //cout<<count1<<" "<<count2<<" "<<count3<<"\n";
        if(count1>count2){
            if(count1>count3){
                //count1
                mres[i] = 1;
            }
            else{
               //count3
                mres[i] = 3;
            }
        }
        else{
           if(count2>count3){
              //count2
              mres[i] = 2;
           }
           else{
               //count3
               mres[i] = 3;
           }
        }

        
        //cout<<mres[i]<<"\n";
        //cout<<"\n=========================================================================\n";
        
    
    }

    // Accuracy calculation.
    int *ggclass,*gggsres,*ggcounter;
    int ccounter[1];
    ccounter[0] = 0;
    cudaMalloc(&ggclass,m*sizeof(int));
    cudaMalloc(&gggsres,m*sizeof(int));
    cudaMalloc(&ggcounter,1*sizeof(int));
    cudaMemcpy(ggclass,fclass,m*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gggsres,mres,m*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(ggcounter,ccounter,1*sizeof(int),cudaMemcpyHostToDevice);
    Accuracy<<<1,m>>>(ggclass,gggsres,ggcounter);
    cudaMemcpy(ccounter,ggcounter,1*sizeof(int),cudaMemcpyDeviceToHost);
    //printf("%d\n",counter[0]);
    float aacc = ccounter[0]*100;
    aacc = aacc/m;

    
    printf("Time taken %f\n",milliseconds1);

    cout<<"Accuracy of KNN after Max Parallelism "<<acc<<"%"<<"\n";
    
    //cout<<"---------------------------------------------\n";
    
    //Free gpu variables

    cudaFree(ggclass);
    cudaFree(gggsres);
    cudaFree(ggcounter);
    cudaFree(gquery2d);
    cudaFree(gdatam);
    cudaFree(gdis);
    cudaFree(gdist);
    cudaFree(gid);
    cudaFree(gid2d);
    cudaFree(gpoint);
    cudaFree(gquery);
    cudaFree(gdata);
    cudaFree(gcounter);
    cudaFree(gclass);
    cudaFree(gsres);
    cudaFree(gres);
    cudaFree(gidd);
    cudaFree(ggdata);

    //Free Cpu variables

    free(data);
    free(fclass);
    free(res);
    free(id);
    free(query);
    free(query2d);
    free(points);
    free(idd);
    free(dis);
    free(id2d);
    free(mres);
    free(dist);
    

    //---------------------------++++++++++++++++++++++++----------------------------
    cudaDeviceSynchronize();
    return 0;
}
