
#include<iostream>
#include<cuda.h>
#include<stdlib.h>
#include<algorithm>
#include<thrust/sort.h>
#include<math.h>
#include<stdio.h>
using namespace std;
struct tree{
    int id;
    int leftid;
    int parent;
    float filter;
    int rightid;
    int pos;
    int startpos;
    int endpos;
}Maptree[30];
__global__ void distance(float *data,float *query,float *dis,int *id,int count,int start){
    int idt = threadIdx.x;
    idt = threadIdx.x+start;
    //printf("%d\n",idt);
    float dist = 0;
    for(int i=1;i<count;i++){
        dist += (data[idt*count+i]-query[i])*(data[idt*count+i]-query[i]);
    }
    dist = sqrt(dist);
    id[threadIdx.x] = data[idt*count+0];
    dis[threadIdx.x] = dist ;
}
__global__ void Accuracy(int *s1,int *s2,int *counter){
    int id = threadIdx.x;
    //printf("%d %d\n",s1[id],s2[id]);
    int x = 1;
    if(s1[id]==s2[id]){
        atomicAdd(&counter[0],x);
    }
}
void KDDpartition(float *index,float *data,int points,int count,int front,int N,int time){
    //cout<<"\n========================================================================\n";
    Maptree[time].id = time;
    int Noofitems = Maptree[time].endpos - Maptree[time].startpos;
    //cout<<Noofitems<<"\n";
    if(Noofitems<points){
        return ;
    }
    float **decide = (float **)malloc(count*sizeof(float*));
    float *mean = (float *)malloc(count*sizeof(float));
    float *var = (float *)malloc(count*sizeof(float));
    for(int i=0;i<count;i++){
        decide[i] = (float *)malloc(N*sizeof(float));
        for(int j=front;j<N;j++){
            decide[i][j] = data[j*count+i];
            mean[i] += decide[i][j]; 
        }
        mean[i] = mean[i]/N;
    }
    for(int i=0;i<count;i++){
        for(int j=front;j<N;j++){
            var[i] +=(decide[i][j]-mean[i])*(decide[i][j]-mean[i]);
        }
        var[i] = var[i]/N;
    }
    float Max = 0;
    int pos = 0;
    for(int i=1;i<count;i++){
        if(Max<var[i]){
            Max = var[i];
            pos = i;
        }
    }
    //cout<<Max<<" "<<pos<<"\n";
    float *cdata = (float *)malloc(N*count*sizeof(float));
    sort(decide[pos]+front,decide[pos]+N);
    for(int i=front;i<N;i++){
        //cout<<decide[pos][i]<<"\t";
    }
    //cout<<"\n";
    int mid = (N-front)/2;
    float Median = decide[pos][front+mid];
    //cout<<mid<<" "<<Median<<"\n";
    int start,last;
    start = Maptree[time].startpos;
    last = Maptree[time].endpos;
    Maptree[time].filter = Median;
    Maptree[time].pos = pos;
    for(int i=front;i<N;i++){
        if(data[i*count+pos]<Median){
            for(int j=0;j<count;j++){
                cdata[start*count+j] = data[i*count+j];
            }
            start++;
        }
        else{
            for(int j=0;j<count;j++){
                cdata[last*count+j] = data[i*count+j];
            }
            last--;
        }
    }
    //cout<<start<<" "<<last<<"\n";
    /*for(int i=front;i<N;i++){
        cout<<i<<"\t";
        for(int j=0;j<count;j++){
            cout<<cdata[i*count+j]<<"\t";
        }
        cout<<"\n";
    }*/
    int left = 2*time;
    int right = 2*time+1;
    Maptree[time].leftid = left;
    Maptree[time].rightid = right;
    Maptree[left].parent = time;
    Maptree[right].parent = time;
    Maptree[left].startpos = front;
    Maptree[left].endpos = last;
    Maptree[right].startpos = start;
    Maptree[right].endpos = Maptree[time].endpos;
    //cout<<Maptree[left].startpos<<" "<<Maptree[left].endpos<<" "<<Maptree[right].startpos<<" "<<Maptree[right].endpos<<"\n";
    for(int i=front;i<N;i++){
        //cout<<i<<"\t";
        for(int j=0;j<count;j++){
            data[i*count+j] = cdata[i*count+j];
            //cout<<data[i*count+j]<<"\t";
        }
        //cout<<"\n";
    }
    KDDpartition(index,data,points,count,Maptree[left].startpos,last+1,left);
    KDDpartition(index,data,points,count,Maptree[right].startpos,Maptree[right].endpos+1,right);
}
void search(float *data,float *query,int points,int count,int N,int m,int time,int k,string s[],string s1[]){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0;
    
    int noofelements = Maptree[time].endpos - Maptree[time].startpos;
    //cout<<noofelements<<"\n";
    int x = time;
    int *fclass = (int *)malloc(m*sizeof(int));
    int *res = (int *)malloc(m*sizeof(int)); 
    float *line = (float *)malloc(count*sizeof(float));
    for(int i=0;i<m;i++){
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
        for(int j=0;j<count;j++){
            line[j] = query[i*count+j];
            //cout<<line[j]<<"\t"; 
        }
        //cout<<"\n";   
        while(noofelements>points){
            int dim = Maptree[x].pos;
            float Median = Maptree[x].filter;
            if(query[i*count+dim]<Median){
                x = Maptree[x].leftid;
            }
            else{
                x = Maptree[x].rightid;
            }
            noofelements = Maptree[x].endpos - Maptree[x].startpos;
        }
        x = Maptree[x].parent;
        int st = Maptree[x].startpos;
        int et = Maptree[x].endpos;
        //cout<<x<<" "<<st<<" "<<et<<"\n";
        float *gdata,*gquery,*dis,*gdis;
        int *id,*gid;
        id = (int *)malloc(N*sizeof(int));
        dis = (float *)malloc(N*sizeof(float));
        float milliseconds = 0;
        cudaEventRecord(start,0);
        cudaMalloc(&gid,N*sizeof(int));
        cudaMalloc(&gdis,N*sizeof(float));
        cudaMalloc(&gdata,N*count*sizeof(float));
        cudaMalloc(&gquery,count*sizeof(float));
        cudaMemcpy(gdata,data,N*count*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(gquery,line,count*sizeof(float),cudaMemcpyHostToDevice);
        //cout<<"\n------------------\n";
        distance<<<1,(et-st)>>>(gdata,gquery,gdis,gid,count,st);
        cudaMemcpy(dis,gdis,N*sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(id,gid,N*sizeof(int),cudaMemcpyDeviceToHost);
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        ms += milliseconds;
        thrust::sort_by_key(dis, dis + (et-st), id);
        int count1,count2,count3;
        count1 = count2 = count3 = 0;
        for(int j=0;j<k;j++){
            //cout<<id[j]<<" "<<dis[j]<<"\n";
            if(id[j]<=50 && id[j]>0){
                count1++;
            }
            if(id[j]>50 && id[j]<=100){
                count2++;
            }
            if(id[j]<=150 && id[j]>100){
                count3++;
            }
        }
        //cout<<"------------------"<<count1<<" "<<count2<<" "<<count3<<"\n";
        if(count1>count2){
            if(count1>count3){
                //count1
                res[i] = 1;
            }
            else{
               //count3
               res[i] = 3;
            }
        }
        else{
           if(count2>count3){
              //count2
               res[i] = 2;
           }
           else{
               //count3
               res[i] = 3;
           }
        }
        x = time;
        noofelements = Maptree[x].endpos - Maptree[x].startpos; 
    }
    int *gclass,*ggsres,*gcounter;
    int counter[1];
    counter[0] = 0;
    cudaMalloc(&gclass,m*sizeof(int));
    cudaMalloc(&ggsres,m*sizeof(int));
    cudaMalloc(&gcounter,1*sizeof(int));
    cudaMemcpy(gclass,fclass,m*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(ggsres,res,m*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(gcounter,counter,1*sizeof(int),cudaMemcpyHostToDevice);
    Accuracy<<<1,m>>>(gclass,ggsres,gcounter);
    cudaMemcpy(counter,gcounter,1*sizeof(int),cudaMemcpyDeviceToHost);
    float acc = counter[0]*100;
    acc = acc/m;
    printf("Total Execution time %f in millisecond\n",ms);
    cout<<"Accuracy of KD tree implementation of KNN "<<acc<<"% \n";

}



////////////////////////
void searchprediction(float *data,float *query,int points,int count,int N,int m,int time,int k,string s[],string s1[]){
    int noofelements = Maptree[time].endpos - Maptree[time].startpos;
    //cout<<noofelements<<"\n";
    int x = time;
    string sf = "";
    while(noofelements>points){
            int dim = Maptree[x].pos;
            float Median = Maptree[x].filter;
            if(query[dim]<Median){
                x = Maptree[x].leftid;
            }
            else{
                x = Maptree[x].rightid;
            }
            noofelements = Maptree[x].endpos - Maptree[x].startpos;
        }
        x = Maptree[x].parent;
        int st = Maptree[x].startpos;
        int et = Maptree[x].endpos;
        //cout<<x<<" "<<st<<" "<<et<<"\n";
        float *gdata,*gquery,*dis,*gdis;
        int *id,*gid;
        id = (int *)malloc(N*sizeof(int));
        dis = (float *)malloc(N*sizeof(float));
        cudaMalloc(&gid,N*sizeof(int));
        cudaMalloc(&gdis,N*sizeof(float));
        cudaMalloc(&gdata,N*count*sizeof(float));
        cudaMalloc(&gquery,count*sizeof(float));
        cudaMemcpy(gdata,data,N*count*sizeof(float),cudaMemcpyHostToDevice);
        cudaMemcpy(gquery,query,count*sizeof(float),cudaMemcpyHostToDevice);
        //cout<<"\n------------------\n";
        distance<<<1,(et-st)>>>(gdata,gquery,gdis,gid,count,st);
        cudaMemcpy(dis,gdis,N*sizeof(float),cudaMemcpyDeviceToHost);
        cudaMemcpy(id,gid,N*sizeof(int),cudaMemcpyDeviceToHost);
        thrust::sort_by_key(dis, dis + (et-st), id);
        int count1,count2,count3;
        count1 = count2 = count3 = 0;
        for(int j=0;j<k;j++){
            //cout<<id[j]<<" "<<dis[j]<<"\n";
            if(id[j]<=50 && id[j]>0){
                count1++;
            }
            if(id[j]>50 && id[j]<=100){
                count2++;
            }
            if(id[j]<=150 && id[j]>100){
                count3++;
            }
        }
        //cout<<"------------------"<<count1<<" "<<count2<<" "<<count3<<"\n";
        if(count1>count2){
            if(count1>count3){
                //count1
                sf = "Iris-setosa";
            }
            else{
               //count3
               sf = "Iris-virginica";
            }
        }
        else{
           if(count2>count3){
              //count2
               sf = "Iris-versicolor";
           }
           else{
               //count3
               sf = "Iris-virginica";
           }
        } 
    cout<<"Predicted output for random point"<<sf<<"\n";
}

////////////////////////





int main(){
    int points = 20;
    int k = 15;
    cout<<"KDD Tree implementation of KNN Algorithm\n";
    FILE *fp;
    int N = 135;
    int count = 0 ; 
    fp = fopen("input.txt","r");
    char ch = ' ';
    while(ch!='\n'){
        ch = getc(fp);
        if(ch==','){
        count++;
        }
    }
    string s[N];
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
            s[i]+=c;
            c = fgetc(fp);
        }
        //cout<<s[i]<<"\n";
    }
    //cout<<"\n=================================================\n";
    int m =15;
    float *query = (float *)malloc(m*count*sizeof(float));
    FILE *op;
    string s1[m];
    op = fopen("test.txt","r");
    for(int i=0;i<m;i++){
        for(int j=0;j<count;j++){
            fscanf(op,"%f",&query[i*count+j]);
            ch = fgetc(op);
            //cout<<query[i*count+j]<<"\t";
        }
        char c;
        c = fgetc(op);
        while(c!='\n'){
            s1[i] += c;
            c = fgetc(op);
        }
        //cout<<s1[i]<<"\n";
    }
    float *index = (float *)malloc(N*2*sizeof(float));
    //Grouping all data
    for(int i=0;i<N;i++){
       index[i*2+0] = 1;
       index[i*2+1] = data[i*count+0];
       //cout<<index[i*2+0]<<" "<<index[i*2+1]<<"\n";
    }
    Maptree[1].id = 1;
    Maptree[1].leftid = 0;
    Maptree[1].filter = 0;
    Maptree[1].rightid = 0;
    Maptree[1].pos = 0;
    Maptree[1].parent = -1;
    Maptree[1].startpos = 0;
    Maptree[1].endpos = 134;
    
    KDDpartition(index,data,points,count,0,N,1);
    //cout<<"\n==============================================================\n";
    /*for(int i=0;i<N;i++){
        for(int j=0;j<count;j++){
           // cout<<data[i*count+j]<<"\t";
        }
        //cout<<"\n";
    }*/
    search(data,query,points,count,N,m,1,k,s,s1);
    
    

    srand(time(0));
    float *point = (float *)malloc(count*sizeof(float));
    for(int j=0;j<count;j++){
        if(j<count-1){
            point[j] = rand()%8;
        }
        else{
            point[j] = rand()%3;
        }
        //cout<<point[j]<<"\t";
    }
    
    searchprediction(data,point,points,count,N,m,1,k,s,s1);
    cudaDeviceSynchronize();
    return 0;
}