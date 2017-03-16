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
#define TILE_SIZE 1024
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
 
/*插入排序*/  
void InsertSort(double array[],int index[],int n,int k)  
{  
	//初始化index数组
	double * q = new double [k];
	for (int i=0;i<k;i++){
		index[i] = i;
	}
	q[0]=array[0];
	double mid=0;
	for (int i=1;i<k;i++){
		q[i]=array[i];
		int j=i;
		while (j>0){
			if(q[j]<q[j-1]){
				mid=q[j];
				q[j]=q[j-1];
				q[j-1]=mid;
				
				mid = index[j];
				index[j] = index[j-1];
				index[j-1] = mid;
				
				j--;
			}
			else
				break;
		}
	}
	for (int i=k;i<n;i++){
		if (array[i]<q[k-1]){
			q[k-1]=array[i];
			index[k-1]=i;

			int j=k-1;
			while (j>0){
				if(q[j]<q[j-1]){
					mid=q[j];
					q[j]=q[j-1];
					q[j-1]=mid;
					
					mid=index[j];
					index[j]=index[j-1];
					index[j-1]=mid;
					
					j--;
				}
				else
					break;
			}
		}
	}
} 	 

__global__ static void sort(int iter,double * distance,double * index,int a,int m,int k){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	double mid;
	double * q = new double [k];	
	if (tid < a){
		int threadNum = blockDim.x * gridDim.x;
		int id = tid;
		for (int i=0;i<k;i++){
			index[id+i*a]=i + iter;
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
				index[id + (k-1)*a]=i + iter;
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
		for (int i=0;i<k;i++){
			distance[id + i*a] = q[i];
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
			mid+=train[i + id * n]*train[i + id * n];
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
	double * gpu_train0, * gpu_train1;
	double * gpu_square0, * gpu_square1;
	double * gpu_test;
	double * gpu_distance;
	double * gpu_distance0, *gpu_distance1;
	double * trainset;
	double * square;
	//double * gpu_index;
	//double * gpu_index0, * gpu_index1;
	
	clock_t start,end;
	
	//创建流
	cudaStream_t stream0;
	cudaStream_t stream1;
	SAFE_CALL(cudaStreamCreate(&stream0));
	SAFE_CALL(cudaStreamCreate(&stream1));

	//stream0
	SAFE_CALL(cudaMalloc((void**)&gpu_train0, TILE_SIZE * n * sizeof(double)));
	SAFE_CALL(cudaMalloc((void**)&gpu_square0, TILE_SIZE * sizeof(double)));
	SAFE_CALL(cudaMalloc((void**)&gpu_distance0, a * TILE_SIZE * sizeof(double)));
	//SAFE_CALL(cudaMalloc((void**)&gpu_index0, a * k * sizeof(double)));
		
	//stream1
	SAFE_CALL(cudaMalloc((void**)&gpu_train1, TILE_SIZE * n * sizeof(double)));
	SAFE_CALL(cudaMalloc((void**)&gpu_square1, TILE_SIZE * sizeof(double)));
	SAFE_CALL(cudaMalloc((void**)&gpu_distance1, a * TILE_SIZE * sizeof(double)));
	//SAFE_CALL(cudaMalloc((void**)&gpu_index1, a * k * sizeof(double)));

	//将CPU内存直接映射到GPU内存空间
	SAFE_CALL(cudaHostAlloc((void**)&trainset, n * m * sizeof(double), cudaHostAllocDefault));
	SAFE_CALL(cudaHostAlloc((void**)&square, m * sizeof(double), cudaHostAllocDefault));
	
	//trainset为矩阵train的转置，便于后期读入数据到显存
	for (int i=0;i<n;i++){
		for (int j=0;j<m;j++)
			trainset[i+j*n]=train[j+i*m];
	}
	
	//将测试集全部拷入显存。未转置
	SAFE_CALL(cudaMalloc((void **) &gpu_test, sizeof(double) * a * b));
	SAFE_CALL(cudaMemcpy(gpu_test,test,sizeof(double) * a * b,cudaMemcpyHostToDevice));
	
	
	//初始化predict_label
	int * predict_label = new int [a];
	for(int i=0;i<a;i++){
		predict_label[i]=0;
	}
	
	
	//labels数组存放训练集中特征点对应的标签
	int ** labels=new int *[a];
	for(int i=0;i<a;i++){
		labels[i]=new int [k];
		for(int j=0;j<k;j++)
			labels[i][j]=0;
	}
		
	int gridSize = 64;
	int blockSize = 512;
	int threadNum = gridSize * blockSize;
	//fprintf(stdout,"Step 1 finished!\n");

	fprintf(stdout,"Start calculating distances:\n");
	start = clock();	
	//调用cublas库进行矩阵乘
	//stream0
	cublasHandle_t handle0;
	cublasCreate(&handle0);
	cublasStatus_t sta0;
	//stream1
	cublasHandle_t handle1;
	cublasCreate(&handle1);
	cublasStatus_t sta1;
	
	//index矩阵初始化
	int index_size =0;
	if (TILE_SIZE>=k){
		index_size = (m%TILE_SIZE)>k ? k:m%TILE_SIZE;
		index_size += (m/TILE_SIZE)*k;
	}
	else{
		index_size=m;
	}
	/*int ** index =new int * [a];
	for (int i=0;i<a;i++){
		index[i] = new int [k];
		for (int j=0;j<k;j++){
			index[i][j]=0;
		}
	}*/
	double ** matr_distance = new double * [a];
	for (int i=0;i<a;i++){
		matr_distance[i] = new double [m];
	}
	double * distance = new double [a * m];
	int i_0;
	double alpha=1.0,beta=0.0;
	for(i_0=0;i_0<m-2*TILE_SIZE;i_0+=2*TILE_SIZE){
		//将训练集(已转置)分片调入显存
		SAFE_CALL(cudaMemcpyAsync(gpu_train0, trainset + i_0*n, TILE_SIZE * n * sizeof(double),cudaMemcpyHostToDevice,stream0));
		SAFE_CALL(cudaMemcpyAsync(gpu_train1, trainset + (i_0+TILE_SIZE)*n, TILE_SIZE * n * sizeof(double),cudaMemcpyHostToDevice,stream1));
		
		//计算square向量
		calculate_square<<<gridSize,blockSize,0,stream0>>>(0,gpu_train0,gpu_square0,TILE_SIZE,n);
		calculate_square<<<gridSize,blockSize,0,stream1>>>(0,gpu_train1,gpu_square1,TILE_SIZE,n);
		
		//设置cublas流
		sta0 = cublasSetStream(handle0,stream0);
		if (sta0 != CUBLAS_STATUS_SUCCESS)	
			fprintf(stdout,"cuBlas error!\n");
		sta0 = cublasDgemm(handle0,CUBLAS_OP_N,CUBLAS_OP_N,a,TILE_SIZE,(n-1),&alpha,gpu_test,a,gpu_train0,n,&beta,gpu_distance0,a);
		if (sta0 != CUBLAS_STATUS_SUCCESS)	
			fprintf(stdout,"cuBlas error!\n");
		sta1 = cublasSetStream(handle1,stream1);
		if (sta1 != CUBLAS_STATUS_SUCCESS)	
			fprintf(stdout,"cuBlas error!\n");
		sta1 = cublasDgemm(handle1,CUBLAS_OP_N,CUBLAS_OP_N,a,TILE_SIZE,(n-1),&alpha,gpu_test,a,gpu_train1,n,&beta,gpu_distance1,a);
		if (sta1 != CUBLAS_STATUS_SUCCESS)	
			fprintf(stdout,"cuBlas error!\n");	
			
		//计算最终的距离	
		calculate_final_distance<<<gridSize,blockSize,0,stream0>>>(0,gpu_distance0,gpu_square0,a,TILE_SIZE);
		calculate_final_distance<<<gridSize,blockSize,0,stream1>>>(0,gpu_distance1,gpu_square1,a,TILE_SIZE);

		//sort
		//sort<<<gridSize,blockSize,0,stream0>>>(i_0,gpu_distance0,gpu_index0,a,TILE_SIZE,k);
		//sort<<<gridSize,blockSize,0,stream1>>>(i_0+TILE_SIZE,gpu_distance1,gpu_index1,a,TILE_SIZE,k);	
		//将计算结果复制到CPU
		SAFE_CALL(cudaMemcpyAsync(distance+i_0*a, gpu_distance0, a * TILE_SIZE * sizeof(double), cudaMemcpyDeviceToHost, stream0));
		//SAFE_CALL(cudaMemcpyAsync(distance+i_0*a, gpu_distance0, a * k * sizeof(double), cudaMemcpyDeviceToHost, stream0));
		//SAFE_CALL(cudaMemcpyAsync(index+i_0*a, gpu_index0, a * k * sizeof(double), cudaMemcpyDeviceToHost, stream0));
		SAFE_CALL(cudaMemcpyAsync(distance+(i_0+TILE_SIZE)*a, gpu_distance1,a * TILE_SIZE * sizeof(double), cudaMemcpyDeviceToHost, stream1));
		//SAFE_CALL(cudaMemcpyAsync(distance+(i_0+TILE_SIZE)*a, gpu_distance1, a * k * sizeof(double), cudaMemcpyDeviceToHost, stream1));
		//SAFE_CALL(cudaMemcpyAsync(index+(i_0+TILE_SIZE)*a, gpu_index1, a * k * sizeof(double), cudaMemcpyDeviceToHost, stream1));
			
		//fprintf(stdout,"The %dth iteration.\n",i_0);
	}
	fprintf(stdout,"i_0:%d\n",i_0);
	if (m%(2*TILE_SIZE)>TILE_SIZE){
		//将训练集(已转置)分片调入显存
		SAFE_CALL(cudaMemcpyAsync(gpu_train0, trainset + i_0*n, TILE_SIZE * n * sizeof(double),cudaMemcpyHostToDevice,stream0));
		SAFE_CALL(cudaMemcpyAsync(gpu_train1, trainset + (i_0+TILE_SIZE)*n, (m%TILE_SIZE) * n * sizeof(double),cudaMemcpyHostToDevice,stream1));
		//计算square向量
		calculate_square<<<TILE_SIZE/blockSize+1,blockSize,0,stream0>>>(0,gpu_train0,gpu_square0,TILE_SIZE,n);
		calculate_square<<<TILE_SIZE/blockSize+1,blockSize,0,stream1>>>(0,gpu_train1,gpu_square1,m%TILE_SIZE,n);
		//设置cublas流
		sta0 = cublasSetStream(handle0,stream0);
		if (sta0 != CUBLAS_STATUS_SUCCESS)	
			fprintf(stdout,"cuBlas error!\n");
		sta0 = cublasDgemm(handle0,CUBLAS_OP_N,CUBLAS_OP_N,a,TILE_SIZE,(n-1),&alpha,gpu_test,a,gpu_train0,n,&beta,gpu_distance0,a);
		if (sta0 != CUBLAS_STATUS_SUCCESS)	
			fprintf(stdout,"cuBlas error!\n");
		sta1 = cublasSetStream(handle1,stream1);
		if (sta1 != CUBLAS_STATUS_SUCCESS)	
			fprintf(stdout,"cuBlas error!\n");
		sta1 = cublasDgemm(handle1,CUBLAS_OP_N,CUBLAS_OP_N,a,(m%TILE_SIZE),(n-1),&alpha,gpu_test,a,gpu_train1,n,&beta,gpu_distance1,a);
		if (sta1 != CUBLAS_STATUS_SUCCESS)	
			fprintf(stdout,"cuBlas error!\n");	
		//计算最终的距离
		calculate_final_distance<<<TILE_SIZE/blockSize+1,blockSize,0,stream0>>>(0,gpu_distance0,gpu_square0,a,TILE_SIZE);
		calculate_final_distance<<<TILE_SIZE/blockSize+1,blockSize,0,stream1>>>(0,gpu_distance1,gpu_square1,a,m%TILE_SIZE);
		
		SAFE_CALL(cudaMemcpyAsync(distance+i_0*a, gpu_distance0, a * TILE_SIZE * sizeof(double), cudaMemcpyDeviceToHost, stream0));
		SAFE_CALL(cudaMemcpyAsync(distance+(i_0+TILE_SIZE)*a, gpu_distance1,a * (m%TILE_SIZE) * sizeof(double), cudaMemcpyDeviceToHost, stream1));
		
	}
	if((m%(2*TILE_SIZE) != 0) && (m%(2*TILE_SIZE)<=TILE_SIZE)){
		//将训练集(已转置)分片调入显存
		SAFE_CALL(cudaMemcpyAsync(gpu_train0, trainset + i_0*n, m%(2*TILE_SIZE) * n * sizeof(double),cudaMemcpyHostToDevice,stream0));
		//计算square向量
		calculate_square<<<gridSize,blockSize,0,stream0>>>(0,gpu_train0,gpu_square0,m%(2*TILE_SIZE),n);
		//设置cublas流
		sta0 = cublasSetStream(handle0,stream0);
		if (sta0 != CUBLAS_STATUS_SUCCESS)	
			fprintf(stdout,"cuBlas error!\n");
		sta0 = cublasDgemm(handle0,CUBLAS_OP_N,CUBLAS_OP_N,a,m%(2*TILE_SIZE),(n-1),&alpha,gpu_test,a,gpu_train0,n,&beta,gpu_distance0,a);
		if (sta0 != CUBLAS_STATUS_SUCCESS)	
			fprintf(stdout,"cuBlas error!\n");
		//计算最终的距离
		calculate_final_distance<<<gridSize,blockSize,0,stream0>>>(0,gpu_distance0,gpu_square0,a,m%(2*TILE_SIZE));
		//复制结果到CPU端
		SAFE_CALL(cudaMemcpyAsync(distance+i_0*a, gpu_distance0, a * (m%(2*TILE_SIZE)) * sizeof(double), cudaMemcpyDeviceToHost, stream0));
	}
	
	SAFE_CALL(cudaFree(gpu_distance0));
	SAFE_CALL(cudaFree(gpu_distance1));
	SAFE_CALL(cudaFree(gpu_square0));
	SAFE_CALL(cudaFree(gpu_square1));
	SAFE_CALL(cudaStreamDestroy(stream0));
	SAFE_CALL(cudaStreamDestroy(stream1));
	SAFE_CALL(cudaDeviceSynchronize());
	//cudaFree(gpu_distance);
	//cudaFree(gpu_index);
	cudaDeviceSynchronize();
	
	end = clock();
	fprintf(stdout,"Using time of calculating distance is : %f\n",float(end-start)/CLOCKS_PER_SEC);
	//将距离矩阵distance转置为matr_distance
	for (int j=0;j<m;j++){
		for (int i=0;i<a;i++){
			matr_distance[i][j] = distance[i+j*a];
		}
		//if (i<20){
		//	fprintf(stdout,"Matrix of distace:%lf %lf %lf %lf\n",matr_distance[i][0],matr_distance[i][32],matr_distance[i][64],matr_distance[i][m-1]);
		//}
	}
	fprintf(stdout,"Sorting...\n");
	start = clock();
	int * index = new int [k];
	//排序
	int mm=0;
	for (int i=0; i<a; i++){
		InsertSort(matr_distance[i],index,m,k);
		//if (i<20){
		//	fprintf(stdout,"Index: %d %d %d %d\n",index[0],index[1],index[2],index[3]);
		//}
		for (int j=0;j<k;j++){
			mm = int(index[j] + (n-1)*m);
			labels[i][j]=int(train[mm]);
		}
	}
	

	//生成预测label
	int * count =new int [nclass]; 
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
	fprintf(stdout,"Using time of sorting : %lf(s)\n",double(end-start)/CLOCKS_PER_SEC);
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
	for (int i=0;i<out;i++){
		fprintf(stdout,"predict label for testset[%d] is %d\n",i,predict_label[i]);
	}
	float accuracy=0.0;
        int right = 0;
	if (b==n){
		for (int i=0;i<a;i++){
			if(predict_label[i] == int(testset[i + (b-1)*a]))
				right++;					                }
		accuracy = float(right)/float(a);
		fprintf(stdout,"Presicion of knnclassifier is:%.2f%%\n",accuracy*100);
	}
	return 0;
}
