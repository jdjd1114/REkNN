#include <mat.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <iostream>
#include "cublas_v2.h"
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
 	

__global__ static void sort(int iter,double * distance,double * index,int a,int m,int k){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
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

__global__ static void calculate_square(int iter,double * train,double * square,int m,int n){
	double mid = 0;
	int tid = blockIdx.x * blockDim.x +threadIdx.x;
	if (tid < m){
		int threadNum = blockDim.x * gridDim.x;
		int id = tid + iter * threadNum;
		for (int i=0;i<n-1;i++){
			mid+=train[id + i*m]*train[id + i*m];
		}
		square[id]=mid;
	}
}

__global__ static void calculate_final_distance(int iter,double * distance,double * square,int a,int m){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < m){
		int threadNum = blockDim.x*gridDim.x;
		int id=tid+iter*threadNum;
		for (int i=0;i<a;i++){
			distance[i+id*a]=square[id]-2*distance[i+id*a];
		}
	}
}

//knn		
int * knn(double * train,int m,int n,double * test,int a,int b,int k,int nclass){
	double * gpu_train,*gpu_test;
	double * gpu_distance;
	double * gpu_index;
	clock_t start,end;

	SAFE_CALL(cudaMalloc((void**) &gpu_train, sizeof(double) * m * n ));
	SAFE_CALL(cudaMemcpy(gpu_train,train,sizeof(double) * m * n,cudaMemcpyHostToDevice));

	SAFE_CALL(cudaMalloc((void **) &gpu_test, sizeof(double) * a * b));
	SAFE_CALL(cudaMemcpy(gpu_test,test,sizeof(double) * a * b,cudaMemcpyHostToDevice));


	
	//初始化predict_label
	int * predict_label = new int [a];
	for(int i=0;i<a;i++){
		predict_label[i]=0;
	}
	
	//double * distance0 = new double [a*m];
	SAFE_CALL(cudaMalloc((void **) &gpu_distance, sizeof(double) * a * m));

	double * gpu_square = new double[m];
	SAFE_CALL(cudaMalloc((void **) &gpu_square, sizeof(double) * m));
	//cudaMemcpy(gpu_distance,distance0,sizeof(double) * a * m,cudaMemcpyHostToDevice);	
	//index数组用于存放特征点的索引位置
	/*int ** index=new int *[a];
	for(int i=0;i<a;i++){
		index[i]=new int [m];
	}
	for(int i=0;i<a;i++)
		for(int j=0;j<m;j++)
			index[i][j]=0;*/
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

	
	for ( i=0;i<m/threadNum;i++){
		calculate_square<<<gridSize,blockSize>>>(i,gpu_train,gpu_square,m,n);
	}
	if (m%threadNum != 0){
		calculate_square<<<gridSize,blockSize>>>(i,gpu_train,gpu_square,m,n);
	}
	//调用cublas库进行矩阵乘
	cublasHandle_t handle;
	cublasCreate(&handle);
	double alpha=1.0,beta=0.0;
	cublasStatus_t sta;
	sta = cublasDgemm(handle,CUBLAS_OP_N,CUBLAS_OP_T,a,m,(n-1),&alpha,gpu_test,a,gpu_train,m,&beta,gpu_distance,a);
	if (sta != CUBLAS_STATUS_SUCCESS)	
		fprintf(stdout,"cuBlas error!\n");

	int i0=0;
	for (i0=0;i0<m/threadNum;i0++){
		calculate_final_distance<<<gridSize,blockSize>>>(i0,gpu_distance,gpu_square,a,m);
	}
	if (m%threadNum != 0){
		calculate_final_distance<<<gridSize,blockSize>>>(i0,gpu_distance,gpu_square,a,m);
	}
	double * distance0 = new double [a*m];
	cudaDeviceSynchronize();
	SAFE_CALL(cudaMemcpy(distance0,gpu_distance,sizeof(double)*a*m,cudaMemcpyDeviceToHost));

	//cudaDeviceSynchronize();
	end = clock();
	double usetime = double(end - start);
	fprintf(stdout,"Calculating distances finished! Usetime:%lf(s)\n",usetime/CLOCKS_PER_SEC);

	for(int i=0;i<20;i++)
		fprintf(stdout,"%lf %lf %lf %lf\n",distance0[i],distance0[a+i],distance0[2*a+i],distance0[3*a+i]);
	
	cudaDeviceSynchronize();
	cudaFree(gpu_test);
	cudaFree(gpu_train);
	cudaFree(gpu_square);
	cudaDeviceSynchronize();

	fprintf(stdout,"CudaFree completed!\n");
	double * index = new double [a*k];
	SAFE_CALL(cudaMalloc((void**) &gpu_index, sizeof(double) * a * k));
	//SAFE_CALL(cudaMemcpy(gpu_index,index,sizeof(double) * a * k,cudaMemcpyHostToDevice));

	start = clock();
	fprintf(stdout,"Start sorting distances:\n");
	
	//cudaDeviceSynchronize();
	//start = clock();
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

	/*for(int i=0;i<a;i++){
		fprintf(stdout,"The %dth iteration of sorting.\n",i);
		index[i]=InsertSort(distance[i],m);
		for(int j=0;j<m;j++){
			labels[i][j]=trainset[index[i][j]][n-1];
		}
		//fprintf(stdout,"%d %d,%d %d\n",index[i][0],labels[i][0],index[i][1],labels[i][1]);
	}*/
	cudaDeviceSynchronize();
	SAFE_CALL(cudaMemcpy(index,gpu_index,sizeof(double) * a * k,cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	cudaFree(gpu_distance);
	cudaFree(gpu_index);
	cudaDeviceSynchronize();

	int *count=new int[nclass];
//	for(int i=0;i<20;i++)
//		fprintf(stdout,"%d %d %d %d\n",int(index[i]),int(index[i+a]),int(index[i+2*a]),int(index[i+3*a]));
	
	int mm=0;
	for (int i=0;i<a;i++){
		for(int j=0;j<k;j++){
			mm = int(index[i+j*a] + (n-1)*m);
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
	
	fprintf(stdout,"number of rows of trainset:%d\n",m);
	fprintf(stdout,"number of columns of trainset:%d\n",n);
	//fprintf(stdout,"Value of train_set[0][4] is:%lf\n",train_set[0][4]);

	//get the number of rows and columns of testset 
	a=mxGetM(test);
	b=mxGetN(test);

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
	/*for (int i=0;i<out;i++){
		fprintf(stdout,"predict label for testset[%d] is %d\n",i,predict_label[i]);
	}*/
	float accuracy=0.0;
        int right = 0;
	if (b==n){
		for (int i=0;i<a;i++){
			if(predict_label[i] == int(testset[i + (b-1)*a]))
				right++;					                }
		accuracy = float(right)/float(a);
	}
	fprintf(stdout,"Presicion of knnclassifier is:%.2f%%\n",accuracy*100);
	return 0;
}
