//-----------------------------CPU Implementation of Knn------------------------------
#include<iostream>
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<chrono>
#include<unistd.h>
using namespace std;
float** distance(float **data,float *points,int N,int m,int count){
    float **arr;
    arr = (float **)malloc(N*sizeof(float*));
    for(int i=0;i<N;i++){
        arr[i] = (float *)malloc(2*sizeof(float));
        arr[i][0] = i;
        float dist = 0;
        for(int j=1;j<count;j++){
             dist += (points[j] - data[i][j])*(points[j] - data[i][j]);
        }
        dist = sqrt(dist);
        arr[i][1] = dist;
    }
    return arr;
}
float** sort(float **arr,int N,int k){
    float** karr = (float **)malloc(k*sizeof(float *));
    for(int i=0;i<k;i++){
        karr[i] = (float *)malloc(2*sizeof(float));
        float min = INT64_MAX;
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
float accuracy(string s[],string s2[],int m){
    int coub = 0;
    float acc;
    for(int i=0;i<m;i++){
        if(s[i]==s2[i]){
            coub++;
        }
    }
    acc = coub*100;
    acc = acc/m;
    return acc;
}
int main(){
    int k=15;
    FILE *fp;
    int count;
    fp = fopen("input.txt","r");
    char ch;
    while(ch!='\n'){
      ch = getc(fp);
      if(ch==','){
         count++;
      }
    }
    //collecting Data points
    float **data;
    int N = 135;
    string s[N];
    data = (float **)malloc(N*sizeof(float *));
    for(int i=0;i<N;i++){
        data[i] = (float *)malloc(count*sizeof(float));
        for(int j=0;j<count;j++){
            fscanf(fp,"%f",&data[i][j]);
            ch = fgetc(fp);
        }
        char c;
        c = getc(fp);
        while(c!='\n'){
            s[i]+=c;
            c = getc(fp);
        }
    }
    // Collecting Test points
    int m =15;
    string s1[m];
    string res[m];
    FILE *op;
    op = fopen("test.txt","r");
    float **query;
    query = (float **)malloc(m*sizeof(float *));
    for(int i=0;i<m;i++){
        query[i] = (float *)malloc(count*sizeof(float));
        for(int j=0;j<count;j++){
            fscanf(op,"%f",&query[i][j]);
            ch = fgetc(op);
        }
        char c;
        c = getc(op);
        while(c!='\n'){
            s1[i]+=c;
            c = getc(op);
        }
    }

    // Testing accuracy of KNN
    auto start = chrono::steady_clock::now();
    for(int i=0;i<m;i++){
        float *points = (float *)malloc(count*sizeof(float));
        for(int j=0;j<count;j++){
            points[j] = query[i][j];
        }
        float** arr = distance(data,points,N,m,count);
        for(int j=0;j<N;j++){
            //cout<<i<<" "<<arr[j][0]<<" "<<arr[j][1]<<"\n";
        }
        //cout<<"\n----------------\n";
        float** minKarr = sort(arr,N,k);
        int count1,count2,count3;
        count1 = count2 = count3 = 0;
        for(int j=0;j<k;j++){
            //cout<<i<<" "<<minKarr[j][0]<<" "<<minKarr[j][1]<<"\n";
            if(s[(int)minKarr[j][0]]=="Iris-setosa"){
                count1++;
            }
            if(s[(int)minKarr[j][0]]=="Iris-versicolor"){
                count2++;
            }
            if(s[(int)minKarr[j][0]]=="Iris-virginica"){
                count3++;
            }
        }
        //cout<<count1<<" "<<count2<<" "<<count3<<"\n";
        //cout<<"\n=================\n";
        if(count1>count2){
             if(count1>count3){
                 //count1
                 res[i] = "Iris-setosa";
             }
             else{
                //count3
                res[i] = "Iris-virginica";
             }
        }
        else{
            if(count2>count3){
               //count2
               res[i] = "Iris-versicolor";
            }
            else{
                //count3
                res[i] = "Iris-virginica";
            }
        }
        
    }
    auto end = chrono::steady_clock::now();
    float acc = accuracy(s1,res,m);
    cout<<"Accuracy of KNN is "<<acc<<"%\n";
    cout<<"Execution time "<<(end-start).count()<<"ns\n";
    srand(time(0));
    // predicting for a random data point
    string prediction;
    float *points = (float *)malloc(count*sizeof(float));
        for(int j=0;j<count;j++){
            if(j<count-1){
            points[j] = rand()%8;
            }
            else{
                points[j] = rand()%3;
            }
        }
        float** arr = distance(data,points,N,m,count);
        for(int j=0;j<N;j++){
            //cout<<i<<" "<<arr[j][0]<<" "<<arr[j][1]<<"\n";
        }
        //cout<<"\n----------------\n";
        float** minKarr = sort(arr,N,k);
        int count1,count2,count3;
        count1 = count2 = count3 = 0;
        for(int j=0;j<k;j++){
            //cout<<i<<" "<<minKarr[j][0]<<" "<<minKarr[j][1]<<"\n";
            if(s[(int)minKarr[j][0]]=="Iris-setosa"){
                count1++;
            }
            if(s[(int)minKarr[j][0]]=="Iris-versicolor"){
                count2++;
            }
            if(s[(int)minKarr[j][0]]=="Iris-virginica"){
                count3++;
            }
        }
        //cout<<count1<<" "<<count2<<" "<<count3<<"\n";
        //cout<<"\n=================\n";
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
    return 0;
}
//-----------------------------------Saurav---------------------------------------