#include <mat.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <iostream>
#include <cublas_v2.h>
#include "cokus.cpp"
#include "cuda_util.h"
#include <cuda_runtime.h>
using namespace std;

#define CUDA_CALL(x) do{ if( (x) != cudaSuccess){\
				printf("Error at %s:%d\n",__FILE__,__LINE__);\
				return EXIT_FAILURE;}}while(0);
bool InitCUDA(){
	int count;
	cudaGetDeviceCount(&count);
	if(count==0){
		fprintf(stderr,"There is no device.\n");
		return false;
	}
	int i;
	for (i =0; i<count;i++){
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop,i)==cudaSuccess){
			if(prop.major>=1){
				break;
			}
		}
	}

	if(i==count){
		fprintf(stderr,"There is no device supporting CUDA 1.x.\n");
		return false;
	}

	cudaSetDevice(i);
	return true;
}
 	//寻找最大值，并返回索引位置
int max(int array[],int n){
	int m=array[0];
	int index=0;
	for(int i=1;i<n;i++){
		if(m<array[i]){
			m=array[i];
			index=i;
		}
        }
	return index;
}


__global__ static void sort(int iter,double * distance,double * index,int a,int m,int n,int k){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	//int mid;
//	double Mid;
	double mid;
	double * q = new double [k];	
	if (tid < a){
		int threadNum = blockDim.x * gridDim.x;
		int id = tid + iter * threadNum;
		for (int i=0;i<k;i++){
			index[id+i*a]=i;
		}

		q[0] = distance[id];
		for (int i=1;i<k;i++){
			q[i]=distance[id + i*a];
			int j=i;
			while(j>0){
				if(q[j]<q[j-1]){
					mid=q[j];
					q[j]=q[j-1];
					q[j-1]=mid;

					mid=index[id + j*a];
					index[id + j*a]=index[id + (j-1)*a];
					index[id + (j-1)*a]=mid;

					j--;
				}	
				else
					break;		
			}
		}

		for (int i=k;i<m;i++){
			if (distance[id + i*a]<q[k-1]){
				q[k-1]=distance[id + i*a];
				index[id + (k-1)*a]=i;
				int j=k-1;
				while(j>0){
					if (q[j]<q[j-1]){
						mid = q[j];
						q[j] = q[j-1];
						q[j-1] =mid;

						mid = index[id + j*a];
						index[id + j*a]=index[id + (j-1)*a];
						index[id +(j-1)*a]=mid;
						j--;
					}
					else
						break;
				}
			}
		}
	}
	return;
}


//并行计算距离
__global__ static void calculate_dis(int iter,double * train,double *test,double *distance,int m,int n,int a,int b){
	double mid;
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < a){
		int threadNum = blockDim.x * gridDim.x;
		int id = tid + iter * threadNum;
		for  (int i=0;i<m;i++){
			mid=0;
			for (int j=0;j<n-1;j++){
				//mid=0;
				mid+=(test[j * a + id]-train[j * m + i])*(test[j * a + id]-train[j * m + i]);
				//distance[i * a + id]=sqrt(mid);
			}
			distance[i * a +id] = sqrt(mid);
		}
	}
	return;
}

//knn		
int * knn(double * train,int m,int n,double * test,int a,int b,int k,int nclass){
	double * gpu_train,*gpu_test;
	double *gpu_distance;
	double * gpu_index;
	clock_t start,end;

	SAFE_CALL(cudaMalloc((void**) &gpu_train, sizeof(double) * m * n ));
	SAFE_CALL(cudaMemcpy(gpu_train,train,sizeof(double) * m * n,cudaMemcpyHostToDevice));

	SAFE_CALL(cudaMalloc((void **) &gpu_test, sizeof(double) * a *b));
	SAFE_CALL(cudaMemcpy(gpu_test,test,sizeof(double) * a * b,cudaMemcpyHostToDevice));

	//初始化predict_label
	int * predict_label = new int [a];
	for(int i=0;i<a;i++){
		predict_label[i]=0;
	}
	
	//distance数组用于存放特征点之间的距离,初始化distance
	double * distance0 = new double [a*m];

	SAFE_CALL(cudaMalloc((void **) &gpu_distance, sizeof(double) * a * m));
	
	//labels数组存放训练集中特征点对应的标签
	int ** labels=new int *[a];
	for(int i=0;i<a;i++){
		labels[i]=new int [k];
	}
	for(int i=0;i<a;i++)
		for(int j=0;j<k;j++)
			labels[i][j]=0;

	//距离排序
	int gridSize = 150;
	int blockSize = 512;
	int threadNum = gridSize * blockSize;
	fprintf(stdout,"Start calculating distances:\n");
	int i;
	
	cudaDeviceSynchronize();
	start = clock();
	for (i=0;i<a/threadNum;i++){
		calculate_dis<<<gridSize , blockSize>>>(i,gpu_train,gpu_test,gpu_distance,m,n,a,b);
	}
	if (a%threadNum != 0){
		calculate_dis<<<gridSize , blockSize>>>(i,gpu_train,gpu_test,gpu_distance,m,n,a,b);
	}
	

	cudaDeviceSynchronize();
	SAFE_CALL(cudaMemcpy(distance0,gpu_distance,sizeof(double)*a*m,cudaMemcpyDeviceToHost));

	cudaDeviceSynchronize();
	end = clock();
	double usetime = double(end - start);
	fprintf(stdout,"Calculating distances finished! Usetime:%lf(s)\n",usetime/CLOCKS_PER_SEC);

	for(int i=0;i<20;i++)
		fprintf(stdout,"%lf %lf %lf %lf\n",distance0[i],distance0[a+i],distance0[2*a+i],distance0[3*a+i]);
	
	cudaDeviceSynchronize();
	cudaFree(gpu_test);
	cudaFree(gpu_train);
	cudaDeviceSynchronize();

	fprintf(stdout,"CudaFree completed!\n");
	double * index = new double [a*k];

	SAFE_CALL(cudaMalloc((void**) &gpu_index, sizeof(double) * a * k));


//	cudaDeviceSynchronize();
	start = clock();
	fprintf(stdout,"Start sorting distances:\n");
	
	
	int ii;
	for (ii=0;ii<a/threadNum;ii++){
		sort<<<gridSize,blockSize>>>(ii,gpu_distance,gpu_index,a,m,n,k);
	}
	if (a%threadNum != 0){
		sort<<<gridSize,blockSize>>>(ii,gpu_distance,gpu_index,a,m,n,k);
	}
	cudaDeviceSynchronize();
	end = clock();
	usetime = double(end - start);
	fprintf(stdout,"Sorting distances finished! Usetime:%lf\n",usetime/CLOCKS_PER_SEC);

	cudaDeviceSynchronize();
	SAFE_CALL(cudaMemcpy(index,gpu_index,sizeof(double) * a * k,cudaMemcpyDeviceToHost));
//	cudaMemcpy(labels,gpu_labels,sizeof(double) * a * m,cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaFree(gpu_distance);
	cudaFree(gpu_index);
	cudaDeviceSynchronize();

	int *count=new int[nclass];
//	for(int i=0;i<20;i++)
//		fprintf(stdout,"%d %d %d %d\n",int(index[i]),int(index[i+a]),int(index[i+2*a]),int(index[i+3*a]));
	
	start = clock();
	int mm=0;
	for (int i=0;i<a;i++){
		for(int j=0;j<k;j++){
			mm = int(index[i+j*a] + (n-1) * m);
			labels[i][j] = int(train [mm]);
		}
	}

	//生成预测label
	for(int i=0;i<a;i++){
		for(int x=0;x<nclass;x++)
			count[x]=0;
		for(int y=0;y<nclass;y++){
			for(int j=0;j<k;j++){
				if(labels[i][j]==(y+1))
					count[y]++;
			}
		}
		int idx=max(count,nclass);
		predict_label[i]=idx+1;
	}
	end = clock();
	usetime = double(end - start);
	fprintf(stdout,"Usetime of generate predit_label:%lf\n",usetime/CLOCKS_PER_SEC);
	
	return predict_label;
}
 
  		
int main(int argc, char * argv[])
{
	if(!InitCUDA()){
		return 0;
	}
	printf("CUDA initialized.\n");

	clock_t start,end;
	int k,a,b,m,n,nclass;
	double *trainset,*testset;
	if(argc!=4){
		fprintf(stderr, "4 input arguments required!");
	}
	MATFile * datamat = matOpen(argv[1], "r");
	mxArray * train = matGetVariable(datamat,"trainset");
	mxArray * test = matGetVariable(datamat,"testset");

	//MATFile * testmat = matOpen(argv[2], "r");
	//mxArray * test = matGetVariable(testmat,"DS");
	
	trainset = (double*)mxGetData(train);
	testset = (double*)mxGetData(test);

	//get the number of rows and columns of trainset
	m=mxGetM(train);
	n=mxGetN(train);
	
//	cudaMemcpy(gpu_m,m,sizeof(int),cudaMemcpyHostToDevice);
//	cudaMemcpy(gpu_n,n,sizeof(int),cudaMemcpyHostToDevice);

	//Matrix train_set
	/*double ** train_set=new double *[m];
	for(int i=0;i<m;i++){
		train_set[i]=new double[n];
	}
	for(int i=0;i<m;i++){
		for(int j=0;j<n;j++){
			train_set[i][j]=trainset[j*m+i];
		}
	}*/
	//trainset = (double **)mxGetData(train);
	fprintf(stdout,"number of rows of trainset:%d\n",m);
	fprintf(stdout,"number of columns of trainset:%d\n",n);
	//fprintf(stdout,"Value of train_set[0][4] is:%lf\n",train_set[0][4]);

	//get the number of rows and columns of testset 
	a=mxGetM(test);
	b=mxGetN(test);
	//Matrix test_set
	/*double ** test_set = new double * [a];
	for (int i=0;i<a;i++){
		test_set[i]=new double [b];
	}
	for (int i=0;i<a;i++){
		for (int j=0;j<b;j++){
			test_set[i][j] = testset[j*a+i];
		}
	}*/
	fprintf(stdout,"Number of rows of testset:%d\n",a);
	fprintf(stdout,"Number of columns of testset:%d\n",b);
	//fprintf(stdout,"Value of test_set[0][3] is:%lf\n",test_set[0][3]);
	if(b!=n && b!=(n-1)){
		fprintf(stderr, "Number of testset's columns should be equal to number of trainset's column!");
	}
	
	//Get the value of k
	k = (int)atoi(argv[2]);
	if(k<=0)
		fprintf(stderr, "Value of k must be greater than zero!");
	
	//Get the number of classes
	nclass = (int)atoi(argv[3]);
	
	//chushihua predict_label
	int * predict_label = new int [a];
	for(int i=0;i<a;i++){
		predict_label[i]=0;
	}
	//fprintf(stdout,"Initialation finished!!!\n");
	start=clock();
	predict_label = knn(trainset,m,n,testset,a,b,k,nclass);
	end=clock();
	double usetime=(double)(end-start);
	//fprintf(stdout,"Predicting labels for testset has finished!\n");
	fprintf(stdout,"Using time of knnclassifier is:%lf(s)\n",usetime/CLOCKS_PER_SEC);
	int out=a;
	if(a>100)
		out=100;
	for (int i=0;i<out;i++){
		fprintf(stdout,"predict label for testset[%d] is %d\n",i,predict_label[i]);
	}	
	double accuracy=0.0;
	int right = 0;
	if (b==n){
		for (int i=0;i<a;i++){
			if(predict_label[i] == int(testset[i + (b-1)*a]))
				right++;
		}
		accuracy=double(right)/double(a);
	}
	fprintf(stdout,"Presicion of knnclassifier is:%.2lf%%\n",100*accuracy);
	return 0;
}
