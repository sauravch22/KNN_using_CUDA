#include<iostream>
#include<stdio.h>
#include<math.h>
using namespace std;
long** distance(long *data,long *points,int N,int m,int count){
    long **arr;
    arr = (long **)malloc(N*sizeof(long*));
    for(int i=0;i<N;i++){
        arr[i] = (long *)malloc(2*sizeof(long));
        arr[i][0] = i;
        float dist = 0;
        for(int j=1;j<count-1;j++){
             dist += (points[j] - data[i*count+j])*(points[j] - data[i*count+j]);
        }
        dist = sqrt(dist);
        arr[i][1] = dist;
    }
    return arr;
}
long** sort(long **arr,int N,int k){
    long** karr = (long **)malloc(k*sizeof(long *));
    for(int i=0;i<k;i++){
        karr[i] = (long *)malloc(2*sizeof(long));
        long min = INT64_MAX;
        //cout<<min<<"\n";
        int pos;
        for(int j=0;j<N;j++){
            if(arr[j][1]<min){
                min = arr[j][1];
                pos = j;
            }
        }
        karr[i][0] = pos;
        karr[i][1] = arr[pos][1];
        arr[pos][1] = INT64_MAX;
    }
    return karr;
}
float Accuracy(long *query,long *res,int m,int count){
    float acc = 0;
    float cow =0 ;
    for(int i=0;i<m;i++){
        if(query[i*count+10]==res[i]){
               cow++;
        }
    }
    acc = cow/m;
    acc = acc*100;
    return acc;
}
int main(){
    int k = 3 ;
    FILE *fp;
    int N = 160000;
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
    long *res = (long *)malloc(m*sizeof(long));
    auto start = chrono::steady_clock::now();
    for(int i=0;i<m;i++)
    {
        long *point = (long *)malloc(count*sizeof(long));
        for(int j=0;j<count;j++){
            point[j] = query[i*count+j];
        }
        long** arr = distance(data,point,N,m,count);
        long** minKarr = sort(arr,N,k);
        //cout<<"\n------------\n";
        int count1,count2;
        count1 = count2 =0 ;
        for(int j=0;j<k;j++){
            //cout<<minKarr[j][0]<<" "<<minKarr[j][1]<<"\n";
            if(data[minKarr[j][0]*count+10]==2){
                count1++;
            }
            if(data[minKarr[j][0]*count+10]==4){
                count2++;
            }
        }
        //cout<<count1<<" "<<count2<<"\n";
        if(count1>count2){
            res[i] = 2;
        }
        else{
            res[i] = 4;
        }
    }
    float acc = Accuracy(query,res,m,count);
    cout<<"Accuracy "<<acc<<" %\n";
    auto end = chrono::steady_clock::now();
    cout<<"Execution time "<<(end-start).count()<<" ns\n";
    return 0;
}