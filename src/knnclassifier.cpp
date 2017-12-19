#include <mat.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <iostream>
using namespace std;


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
	q[0]=array[0];
	for (int i=1;i<k;i++){
		q[i]=array[i];
		int j=i;
		while (j>0){
			if(q[j]<q[j-1]){
				swap(q[j],q[j-1]);
				swap(index[j],index[j-1]);
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
					swap(q[j],q[j-1]);
					swap(index[j],index[j-1]);
					j--;
				}
				else
					break;
			}
		}
	}
}

//计算距离函数
void calculate_distance(double ** tr,int tr_size,double ** te,int te_size,double ** distance,int n){
	for (int i=0;i<te_size;i++){
		for(int j=0;j<tr_size;j++){
			for (int x=0;x<n;x++){
				distance[i][j]+=(te[i][x]-tr[j][x])*(te[i][x]-tr[j][x]);
			}
			distance[i][j]=sqrt(distance[i][j]);
		}
	}
}

//knn
int * knn(double ** train,int m,int n,double ** test,int a,int b,int k,int nclass){
	//初始化predict_label
	int * predict_label = new int [a];
	for(int i=0;i<a;i++){
		predict_label[i]=0;
	}

	//distance数组用于存放特征点之间的距离,初始化distance
	double ** distance=new double *[a];
	for(int i=0;i<a;i++){
		distance[i]=new double [m];
	}
	for(int i=0;i<a;i++)
		for(int j=0;j<m;j++)
			distance[i][j]=0;

	//index数组用于存放特征点的索引位置
	int ** index=new int *[a];
	for(int i=0;i<a;i++){
		index[i]=new int [k];
	}
	for(int i=0;i<a;i++)
		for(int j=0;j<k;j++)
			index[i][j]=j;

	//labels数组存放训练集中特征点对应的标签
	int ** labels=new int *[a];
	for(int i=0;i<a;i++){
		labels[i]=new int [k];
	}
	for(int i=0;i<a;i++)
		for(int j=0;j<k;j++)
			labels[i][j]=0;

	//计算距离
	calculate_distance(train,m,test,a,distance,n-1);
	/*for(int i=0;i<a;i++){
		for(int j=0;j<m;j++){
			for(int x=0;x<b;x++){
				distance[i][j]+=(test[i][x]-train[j][x])*(test[i][x]-train[j][x]);
			}
			distance[i][j]=sqrt(distance[i][j]);

		}
		//fprintf(stdout,"%lf %lf %lf %lf\n",distance[i][0],distance[i][1],distance[i][2],distance[i][3]);
	}*/
	//距离排序
	for(int i=0;i<a;i++){
		InsertSort(distance[i],index[i],m,k);
		for(int j=0;j<k;j++){
			labels[i][j]=train[index[i][j]][n-1];
		}
		//fprintf(stdout,"%d %d,%d %d\n",index[i][0],labels[i][0],index[i][1],labels[i][1]);
	}

	int *count=new int[nclass];

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

	//Matrix train_set
	double ** train_set=new double *[m];
	for(int i=0;i<m;i++){
		train_set[i]=new double[n];
	}
	for(int i=0;i<m;i++){
		for(int j=0;j<n;j++){
			train_set[i][j]=trainset[j*m+i];
		}
	}
    cout << "==========================================\n";
	fprintf(stdout, "Training Set\n");
    fprintf(stdout, " row : %d    col : %d\n", m, n);

	//get the number of rows and columns of testset
	a=mxGetM(test);
	b=mxGetN(test);

	//Matrix test_set
	double ** test_set = new double * [a];
	for (int i=0;i<a;i++){
		test_set[i]=new double [b];
	}
	for (int i=0;i<a;i++){
		for (int j=0;j<b;j++){
			test_set[i][j] = testset[j*a+i];
		}
	}

	fprintf(stdout,"\nTesting Set\n");
	fprintf(stdout," row : %d    col : %d\n", a, b);
    cout << "==========================================\n";
	if(b!=n && b!=(n-1)){
		fprintf(stderr, "Number of testset's columns should be equal to number of trainset's column!\n");
	}

	//Get the value of k
	k = (int)atoi(argv[2]);
	if(k<=0)
		fprintf(stderr, "Value of k must be greater than zero!");

	//Get the number of classes
	nclass = (int)atoi(argv[3]);

	//chushihua predict_label
	int * predict_label = new int [a]();

	start=clock();
	predict_label = knn(train_set,m,n,test_set,a,b,k,nclass);
	end=clock();
	double usetime=(double)(end-start);
	fprintf(stdout,"Done. Using time of knnclassfier is : %.3lf(s)\n", usetime/CLOCKS_PER_SEC);
	
    float accuracy = 0.0;
	int right = 0;
	if(b == n){
		for (int i=0;i<a;i++){
			if(predict_label[i] == test_set[i][b-1])
				right++;
		}
		accuracy = float(right)/float(a);
	}
	fprintf(stdout,"Precision of knnclassifier is:%.2f%%\n",accuracy*100);
	return 0;
}
