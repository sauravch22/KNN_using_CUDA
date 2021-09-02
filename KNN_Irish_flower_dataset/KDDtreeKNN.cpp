//-----------------------------CPU Implementation Of KDtree KNN--------------------------------
//Add a member in the structure to store the pos of dim for each time filteration.
//Mistake in sort function in durring assignment of pos line 51
#include<iostream>
#include<algorithm>
#include<stdio.h>
#include<math.h>
#include<stdlib.h>
#include<chrono>
#include<unistd.h>
using namespace std;
struct tree{
     int id;
     int leftid;
     int parent;
     float filter;
     int righid;
     int pos;
     int startpos;
     int endpos;
}Maptree[30];

//Brute Force Way of Calculating Distance but the Sample space reduce due to the preprocessing done by Kd tree..

float ** distance(float *data,float *query,int start,int end,int count,int N,int pos){
    int sizearr = end - start;
    sizearr = sizearr+1;
    float **arr =(float **)malloc(sizearr*sizeof(float*));
    int c=0;
    for(int i=start;i<=end;i++){
        arr[c] = (float *)malloc(2*sizeof(float));
        arr[c][0] = data[i*count+0];
        float dist = 0;
        for(int j=1;j<count;j++){
           dist += (query[pos*count+j]-data[i*count+j])*(query[pos*count+j]-data[i*count+j]);
        }
        dist = sqrt(dist);
        arr[c][1] = dist;
        c++;
    }
    return arr;
}

//distance function for prediction point 

float ** distance(float *data,float *query,int start,int end,int count,int N){
    int sizearr = end - start;
    sizearr = sizearr+1;
    float **arr =(float **)malloc(sizearr*sizeof(float*));
    int c=0;
    for(int i=start;i<=end;i++){
        arr[c] = (float *)malloc(2*sizeof(float));
        arr[c][0] = data[i*count+0];
        float dist = 0;
        for(int j=1;j<count;j++){
           dist += (query[j]-data[i*count+j])*(query[j]-data[i*count+j]);
        }
        dist = sqrt(dist);
        arr[c][1] = dist;
        c++;
    }
    return arr;
}

//Sorting K nearest neighbour based on distance.

float** sort(float **arr,int N,int k){
    float** karr = (float **)malloc(k*sizeof(float *));
    for(int i=0;i<k;i++){
        karr[i] = (float *)malloc(2*sizeof(float));
        float min = INT64_MAX;
        //cout<<min<<"\n";
        int pos;
        for(int j=0;j<N+1;j++){
            if(arr[j][1]<min){
                min = arr[j][1];
                pos = j;
            }
        }
        karr[i][0] = arr[pos][0];
        karr[i][1] = arr[pos][1];
        arr[pos][1] = INT64_MAX;
    }
    return karr;
}

//Measuring KNN performance

float accuracy(string s[],string s2[],int m){
    float coub = 0;
    float acc;
    for(int i=0;i<m;i++){
        //cout<<s[i]<<" "<<s2[i]<<"\n";
        if(s[i].compare(s2[i])==0){
            coub++;
        }
    }
    //cout<<coub;
    acc = coub/m;
    acc = acc*100;
    return acc;
}

//Construction of the KD tree So that Sample space decrease for the no of points needed to compute distance.

void KDDpartition(float *index,float *data,int points,int count,int front,int N,int time){
    //cout<<"=================================================\n";
    Maptree[time].id = time;
    int Noofitems = Maptree[time].endpos-Maptree[time].startpos;
    //cout<<Noofitems<<"\n";
    if(Noofitems<points){
        return;
    }
    float **decide = (float **)malloc(count*sizeof(float*));
    float *mean = (float *)malloc(count*sizeof(float));
    float *var = (float *)malloc(count*sizeof(float));
    for(int i=0;i<count;i++){
        decide[i] = (float *)malloc(N*sizeof(float));
        for(int j=front;j<N;j++){
            decide[i][j] = data[j*count+i];
            mean[i] +=decide[i][j];
        }
        mean[i] = mean[i]/N;
    }
    for(int i=0;i<count;i++)
    {
        for(int j=front;j<N;j++){
            var[i] += (decide[i][j]-mean[i])*(decide[i][j]-mean[i]);
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
    /*for(int i=front;i<N;i++){
        //cout<<decide[pos][i]<<"\t";
    }*/
    //cout<<"\n";
    int mid = (N-front)/2;
    //cout<<mid<<"\n";
    float Median = decide[pos][front+mid];
    //cout<<Median<<"\n";
    //Update Maptree through I add it on function parameter;
    int start,last;
    start = Maptree[time].startpos;
    last = Maptree[time].endpos;
    //cout<<start<<" "<<last<<"\n";
    //cout<<Median<<" "<<pos<<"\n";
    Maptree[time].filter = Median;
    Maptree[time].pos = pos;
    for(int i=front;i<N;i++){
        if(data[i*count+pos]<Median){
            //cout<<"left"<<" "<<start<<" "<<i<<"\n";
            for(int j=0;j<count;j++){
                cdata[start*count+j] = data[i*count+j];
            }
            start++;
        }
        else{
            //cout<<"right"<<" "<<last<<" "<<i<<"\n";
            for(int j=0;j<count;j++){
                cdata[last*count+j] = data[i*count+j];
            }
            last--;
        }
    }
    //cout<<"*"<<" "<<last<<" "<<start<<"\n";
    int left = 2*time;
    int right = 2*time+1;
    Maptree[time].leftid = left;
    Maptree[time].righid = right;
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

//Searching for the Sample space which is nearer to the train points.

void KDDsearch(float *data,float *query,int points,int count,int N,int m,int time,int k,string s[],string s2[]){
    int noofelements = Maptree[time].endpos-Maptree[time].startpos;
    string res[m];
    //cout<<noofelements<<"\n";
    int x = time;
    for(int i=0;i<m;i++){
        while(noofelements>points){
            int dim = Maptree[x].pos;
            float Median = Maptree[x].filter;
            if(query[i*count+dim]<Median){
                 x = Maptree[x].leftid;
            }
            else{
                 x = Maptree[x].righid;
            }
            noofelements = Maptree[x].endpos - Maptree[x].startpos;
        }
        x = Maptree[x].parent;
        int st = Maptree[x].startpos;
        int et = Maptree[x].endpos;
        //cout<<"\n"<<x<<"\t"<<st<<" "<<et<<"\n";
        //write the next line of code.
        //cout<<"\n------------------\n";
        float **arr = distance(data,query,st,et,count,N,i);
        /*for(int i=0;i<=(et-st);i++){
            //cout<<i<<" "<<arr[i][0]<<"\t"<<arr[i][1]<<"\n";
        }*/
        float **minarr = sort(arr,(et-st),k);
        int count1,count2,count3;
        count1 = count2 = count3 = 0;
        //cout<<"\n===================\n";
        for(int i=0;i<k;i++){
            //cout<<i<<" "<<minarr[i][0]<<"\t"<<minarr[i][1]<<"\n";
            int z = minarr[i][0];
            if(z>0 && z<=50){
                 count1++;
            }
            if(z>50 && z<=100){
                 count2++;
            }
            if(z>100 && z<=150){
                 count3++;
            }
        }
        //cout<<"--------------"<<count1<<" "<<count2<<" "<<count3<<"\n";
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
        //cout<<res[i]<<"\n";

        x = time;
        noofelements = Maptree[x].endpos - Maptree[x].startpos;
    }
    float acc = accuracy(s2,res,m);
    cout<<"Accuracy of KNN is "<<acc<<" %\n";
}


//prediction function......

string KDDpredictionsearch(float *data,float *query,int points,int count,int N,int m,int time,int k,string s[],string s2[]){
    int noofelements = Maptree[time].endpos-Maptree[time].startpos;
    string res;
    //cout<<noofelements<<"\n";
    int x = time;
    
        while(noofelements>points){
            int dim = Maptree[x].pos;
            float Median = Maptree[x].filter;
            if(query[dim]<Median){
                 x = Maptree[x].leftid;
            }
            else{
                 x = Maptree[x].righid;
            }
            noofelements = Maptree[x].endpos - Maptree[x].startpos;
        }
        x = Maptree[x].parent;
        int st = Maptree[x].startpos;
        int et = Maptree[x].endpos;
        //cout<<"\n"<<x<<"\t"<<st<<" "<<et<<"\n";
        //write the next line of code.
        //cout<<"\n------------------\n";
        float **arr = distance(data,query,st,et,count,N);
        /*for(int i=0;i<=(et-st);i++){
            //cout<<i<<" "<<arr[i][0]<<"\t"<<arr[i][1]<<"\n";
        }*/
        float **minarr = sort(arr,(et-st),k);
        int count1,count2,count3;
        count1 = count2 = count3 = 0;
        //cout<<"\n===================\n";
        for(int i=0;i<k;i++){
            //cout<<i<<" "<<minarr[i][0]<<"\t"<<minarr[i][1]<<"\n";
            int z = minarr[i][0];
            if(z>0 && z<=50){
                 count1++;
            }
            if(z>50 && z<=100){
                 count2++;
            }
            if(z>100 && z<=150){
                 count3++;
            }
        }
        //cout<<"--------------"<<count1<<" "<<count2<<" "<<count3<<"\n";
        if(count1>count2){
             if(count1>count3){
                 //count1
                 res = "Iris-setosa";
             }
             else{
                //count3
                res = "Iris-virginica";
             }
        }
        else{
            if(count2>count3){
               //count2
               res = "Iris-versicolor";
            }
            else{
                //count3
                res = "Iris-virginica";
            }
        }
    return res;    
}

//Main Function of the program .

int main(){
    int points =20;//Sample space suitable for these train Space
    //struct tree Maptree[30];
    int k=15;//No of Nearest neighbour taken into consideration
    cout<<"KD tree implementation of KNN\n";
    FILE *fp;
    fp = fopen("input.txt","r");
    int N = 135;
    int count ;
    char ch;
    while(ch!='\n'){
      ch = getc(fp);
      if(ch==','){
         count++;
      }
    }
    string s[N];
    //cout<<count;

    //Reading training data
    //data array store training data.
    float *data = (float *)malloc(N*count*sizeof(float));
    for(int i=0;i<N;i++){
        for(int j=0;j<count;j++){
            fscanf(fp,"%f",&data[i*count+j]);
            ch = fgetc(fp);
            //cout<<data[i*count+j]<<" ";
        }
        char c;
        c = getc(fp);
        while(c!='\n'){
            s[i] +=c;
            c = getc(fp);
        }
        //cout<<s[i]<<"\n";
    }
    int m = 15;
    string s1[m];
    FILE *op;
    op = fopen("test.txt","r");
    float *query = (float *)malloc(m*count*sizeof(float));

    //Reading testing data.

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

    //Initializing the KD tree. 

    Maptree[1].id= 1;
    Maptree[1].leftid = 0;
    Maptree[1].filter = 0;
    Maptree[1].righid = 0;
    Maptree[1].pos = 0;
    Maptree[1].parent = -1;
    Maptree[1].startpos = 0;
    Maptree[1].endpos = 134;

    //Calling the function partition.
    
    KDDpartition(index,data,points,count,0,N,1);
    //cout<<"\n=====================================================\n";
    /*for(int i=0;i<N;i++){
        //cout<<i<<"\t";
        for(int j=0;j<count;j++){
            //cout<<data[i*count+j]<<"\t";
        }
        //cout<<"\n";
    }*/

    //Searching for the suitable Sample Space.
    auto start = chrono::steady_clock::now();
    KDDsearch(data,query,points,count,N,m,1,k,s,s1);
    auto end = chrono::steady_clock::now();
    cout<<"Execution time "<<(end-start).count()<<"ns\n";
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
    //cout<<"\n";
    string prediction = KDDpredictionsearch(data,point,points,count,N,m,1,k,s,s1);
    cout<<"Prediction result for a random point "<<prediction<<"\n";
    return 0;
}
//---------------------------------Saurav-------------------------------------------