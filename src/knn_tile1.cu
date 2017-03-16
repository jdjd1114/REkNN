#include <mat.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <thrust/sort.h>
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

//查找最小值，并返回索引
int min(double array[],int n){
	double m = array[0];
	int ind=0; 
	for (int i=0;i<n;i++){
		if(m>array[i]){
			m=array[i];
			ind=i;
		} 
	}
	return ind;
}

 //全局查找KNN
void global_knn(double * train, double * distance, double * index, int * labels, int * predict_label, int a, int m, int n, int k, int nclass){
	 int tile_num = (m%TILE_SIZE==0) ? (m/TILE_SIZE):(m/TILE_SIZE + 1);//计算分片的个数
	 int rank;
	 int * count =new int [nclass]; 
	 for (int i=0;i<a;i++){
		 rank = 0;
		 double * q = new double [tile_num];//指向当前每个局部最优的最小值
		 int * p = new int [tile_num];//指向当前最小值在index矩阵（或distance矩阵）中的索引
		
		 //初始化count数组
		 for (int j=0;j<nclass;j++){
		 	count[j] = 0;
		 }
		 //初始化，指针指向每个分片的第一个距离
		 for (int j=0;j<tile_num;j++){
			 p[j] = i + j * k * a; //第i行第j*k列，每个分片的返回值有k列
			 q[j] = distance[p[j]];
		 }

		 while(rank<k){
			int ind = min(q,tile_num);//ind为当前最小值的索引位置
			int row = int(index[p[ind]]);
			labels[i + rank * a] = int(train[row + (n-1)*m]);

			for (int x=0;x<nclass;x++){
				if(labels[i + rank *a] == (x+1)){
					count[x] ++;
				}
			}
			rank ++;
			p[ind] += a;//指针挪至下一列
			q[ind] = distance[p[ind]];
		 }
		 
		/*if (i<10){
			fprintf(stdout,"labels of knn : %d %d %d %d\n",labels[i],labels[i+1*a],labels[i+2*a],labels[i+3*a]);
			fprintf(stdout,"count of labels : %d %d %d %d\n",count[0],count[1],count[2],count[3]);
		}*/
		int idx=max(count,nclass);
		predict_label[i]=idx+1;
	 }
 }
 
/*排序，得到局部的k近邻*/  
__global__ static void local_sort(int iter,double * distance,double * index,int a,int m,int k){
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	double mid = 0;
	if(tid < a){
		for (int i=0;i<k;i++){
			//q[i] = distance[tid + i * a];
			index[tid + i * a] = i + iter;
		}
		
		//对前k个距离进行冒泡排序
		for (int i=0;i<k-1;i++){
			for (int j=i+1;j<k;j++){
				if(distance[tid + i * a] > distance[tid + j * a]){
					mid = distance[tid + i*a];
					distance[tid + i*a] = distance[tid + j*a];
					distance[tid + j*a] = mid;

					mid = index[tid + i * a];
					index[tid + i*a] = index[tid + j*a];
					index[tid + j*a] = mid;	
				}
			}
		}
		//将后m-k个数插入到优先队列当中
		for (int i=k;i<m;i++){
			if (distance[tid + i*a] < distance[tid + (k-1)*a]){
				distance[tid + (k-1)*a] = distance[tid + i*a];
				index[tid +(k-1)*a] = i + iter;
				for (int j=k-1;j>0;j--){
					if(distance[tid + j*a] < distance[tid + (j-1)*a]){
						mid = distance[tid + (j-1)*a];
						distance[tid + (j-1)*a] = distance[tid + j*a];
						distance[tid + j*a] = mid;

						mid = index[tid +(j-1)*a];
						index[tid + (j-1)*a] = index[tid + j*a];
						index[tid +j*a] = mid;
					}
					else{
						break;
					}
				}
			}
		}
	}
}

//计算训练集的模2值
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

//计算相对距离
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
	//double * gpu_distance;
	double * gpu_distance0, *gpu_distance1;
	double * trainset;
	double * square;
	//double * gpu_index;
	double * gpu_index0, * gpu_index1;
	
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
	SAFE_CALL(cudaMalloc((void**)&gpu_index0, a * k * sizeof(double)));
		
	//stream1
	SAFE_CALL(cudaMalloc((void**)&gpu_train1, TILE_SIZE * n * sizeof(double)));
	SAFE_CALL(cudaMalloc((void**)&gpu_square1, TILE_SIZE * sizeof(double)));
	SAFE_CALL(cudaMalloc((void**)&gpu_distance1, a * TILE_SIZE * sizeof(double)));
	SAFE_CALL(cudaMalloc((void**)&gpu_index1, a * k * sizeof(double)));

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
	
	
	//labels数组存放训练集中k近邻的特征点对应的标签
	int * labels=new int [a * k];
		
	int gridSize = 64;
	int blockSize = 512;
	//int threadNum = gridSize * blockSize;
	//fprintf(stdout,"Step 1 finished!\n");

	fprintf(stdout,"Start calculating distance matrix...\n");
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
		index_size = (m%TILE_SIZE)<k ? (m%TILE_SIZE):k;
		index_size += (m/TILE_SIZE)*k;
		//fprintf(stdout,"m%TILE_SIZE = %d \n",(m%TILE_SIZE));
	}
	else{
		index_size=m;
	}
	//fprintf(stdout,"index size is : %d \n",index_size);
	/*int ** index =new int * [a];
	for (int i=0;i<a;i++){
		index[i] = new int [k];
		for (int j=0;j<k;j++){
			index[i][j]=0;
		}
	}*/
	double ** matr_distance = new double * [a];
	for (int i=0;i<a;i++){
		matr_distance[i] = new double [index_size];
	}
	double * distance = new double [a * index_size];
	double * index = new double [a * index_size];
	int i_0;
	int i_1 = 0;
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

		//排序
		local_sort<<<gridSize,blockSize,0,stream0>>>(i_0,gpu_distance0,gpu_index0,a,TILE_SIZE,k);
		local_sort<<<gridSize,blockSize,0,stream1>>>(i_0+TILE_SIZE,gpu_distance1,gpu_index1,a,TILE_SIZE,k);
	
		//将计算结果复制到CPU
		//SAFE_CALL(cudaMemcpyAsync(distance+i_0*a, gpu_distance0, a * TILE_SIZE * sizeof(double), cudaMemcpyDeviceToHost, stream0));
		SAFE_CALL(cudaMemcpyAsync(distance+i_1 * a, gpu_distance0, a * k * sizeof(double), cudaMemcpyDeviceToHost, stream0));
		SAFE_CALL(cudaMemcpyAsync(index+i_1 * a, gpu_index0, a * k * sizeof(double), cudaMemcpyDeviceToHost, stream0));
		
		//SAFE_CALL(cudaMemcpyAsync(distance+(i_0+TILE_SIZE)*a, gpu_distance1,a * TILE_SIZE * sizeof(double), cudaMemcpyDeviceToHost, stream1));
		SAFE_CALL(cudaMemcpyAsync(distance+(i_1 + k)*a, gpu_distance1, a * k * sizeof(double), cudaMemcpyDeviceToHost, stream1));
		SAFE_CALL(cudaMemcpyAsync(index+(i_1 + k) * a, gpu_index1, a * k * sizeof(double), cudaMemcpyDeviceToHost, stream1));
			
		//fprintf(stdout,"The %dth iteration.\n",i_0/TILE_SIZE);
		i_1 += 2*k;
	}
	//fprintf(stdout,"i_0:%d\n",i_0);
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
		
		int mm = 0;
		mm = (m%TILE_SIZE) > k ? k:(m%TILE_SIZE);
		//排序
		local_sort<<<gridSize,blockSize,0,stream0>>>(i_0,gpu_distance0,gpu_index0,a,TILE_SIZE,k);
		local_sort<<<gridSize,blockSize,0,stream1>>>(i_0+TILE_SIZE,gpu_distance1,gpu_index1,a,(m%TILE_SIZE),mm);
		
		//将计算结果复制到CPU
		//SAFE_CALL(cudaMemcpyAsync(distance+i_0*a, gpu_distance0, a * TILE_SIZE * sizeof(double), cudaMemcpyDeviceToHost, stream0));
		SAFE_CALL(cudaMemcpyAsync(distance+ i_1 * a, gpu_distance0, a * k * sizeof(double), cudaMemcpyDeviceToHost, stream0));
		SAFE_CALL(cudaMemcpyAsync(index+ i_1 * a, gpu_index0, a * k * sizeof(double), cudaMemcpyDeviceToHost, stream0));
		
		//SAFE_CALL(cudaMemcpyAsync(distance+(i_0+TILE_SIZE)*a, gpu_distance1,a * TILE_SIZE * sizeof(double), cudaMemcpyDeviceToHost, stream1));
		SAFE_CALL(cudaMemcpyAsync(distance+(i_1 + k)*a, gpu_distance1, a * mm * sizeof(double), cudaMemcpyDeviceToHost, stream1));
		SAFE_CALL(cudaMemcpyAsync(index+(i_1 + k)*a, gpu_index1, a * mm * sizeof(double), cudaMemcpyDeviceToHost, stream1));
		
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
		
		int mm = (m%TILE_SIZE) > k ? k:(m%TILE_SIZE);
		//排序
		local_sort<<<gridSize,blockSize,0,stream0>>>(i_0,gpu_distance0,gpu_index0,a,(m%TILE_SIZE),mm);
		
		//复制结果到CPU端
		SAFE_CALL(cudaMemcpyAsync(distance+i_1 * a, gpu_distance0, a * mm * sizeof(double), cudaMemcpyDeviceToHost, stream0));
		SAFE_CALL(cudaMemcpyAsync(index+i_1 * a, gpu_index0, a * mm * sizeof(double), cudaMemcpyDeviceToHost, stream0));
	}
	
	SAFE_CALL(cudaFree(gpu_distance0));
	SAFE_CALL(cudaFree(gpu_distance1));
	SAFE_CALL(cudaFree(gpu_square0));
	SAFE_CALL(cudaFree(gpu_square1));
	SAFE_CALL(cudaFree(gpu_index0));
	SAFE_CALL(cudaFree(gpu_index1));
	SAFE_CALL(cudaStreamDestroy(stream0));
	SAFE_CALL(cudaStreamDestroy(stream1));
	SAFE_CALL(cudaDeviceSynchronize());
	cudaDeviceSynchronize();
	
	end = clock();
	fprintf(stdout,"Time costed to calculate distance matrix: %f s\n",float(end-start)/CLOCKS_PER_SEC);
	/*
	for (int i=0;i<10;i++){
		fprintf(stdout,"Sorted distances: %lf %lf,%lf %lf,%lf %lf\n", distance[i], distance[i + 1*a],distance[i+k*a],distance[i+(k+1)*a],distance[i+2*k*a],distance[i+(2*k+1)*a],distance[i+(index_size-2)*a],distance[i+(index_size-1)*a]);
		fprintf(stdout,"Sorted index: %d %d,%d %d,%d %d,%d %d\n",int(index[i]), int(index[i + 1*a]), int(index[i+k*a]), int(index[i+(k+1)*a]), int(index[i+2*k*a]), int(index[i+(2*k+1)*a]), int(index[i+(index_size-2)*a]), int(index[i+(index_size-1)*a]));
	}*/
	start = clock();
	global_knn(train,distance,index,labels,predict_label,a,m,n,k,nclass);
	end = clock();
	fprintf(stdout,"Time costed to sort distances : %lf s\n",double(end-start)/CLOCKS_PER_SEC);
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
	
	fprintf(stdout,"Training set\n row:%d    ",m);
	fprintf(stdout,"cloumn:%d\n",n);
	//fprintf(stdout,"Value of train_set[0][4] is:%lf\n",train_set[0][4]);

	//get the number of rows and columns of testset 
	a=mxGetM(test);
	b=mxGetN(test);

	fprintf(stdout,"Testing set\n row:%d    ",a);
	fprintf(stdout,"column:%d\n",b);
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
	fprintf(stdout,"Processing time of knnclassifier is:%lf s\n",usetime/CLOCKS_PER_SEC);
	int out=a;
	if(a>10)
		out=10;
	for (int i=0;i<out;i++){
		if(i%2 == 0)
			fprintf(stdout,"predict label for testset[%d] is %d.   ",i,predict_label[i]);
		if(i%2 != 0)
			fprintf(stdout,"predict label for testset[%d] is %d.\n",i,predict_label[i]);
	}
	float accuracy=0.0;
        int right = 0;
	if (b==n){
		for (int i=0;i<a;i++){
			if(predict_label[i] == int(testset[i + (b-1)*a]))
				right++;					                }
		accuracy = float(right)/float(a);
		fprintf(stdout,"Presicion of knnclassifier is:%.2f%%.\n",accuracy*100);
	}
	return 0;
}
