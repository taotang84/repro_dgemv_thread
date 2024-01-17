#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#define __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#ifndef EPS_DOUBLE
#define EPS_DOUBLE 2.0e-52
#endif
#include"tools.h"
#include"cblas.h"
#include <inttypes.h>
#include <sys/time.h>
#include <unistd.h>
#include <math.h>
#include <getopt.h>
#include <ctype.h>
#include<time.h>
#include"/home/xks/OPENBLAS-EXT/OpenBLAS-0.3.14/kernel/simd/intrin.h"
#include"/home/xks/OPENBLAS-EXT/OpenBLAS-0.3.14/common.h"
void functional_test(char order, char trans, long m, long n, long kl, long ku, double alpha, long lda, long inc_x, double beta, long inc_y, double epsilon);
void cblas_repro_dgbmv_d(char order, char trans,long m, long n, long ku, long kl, double alpha, double *b, long ldb,double *x, long incx, double beta, double *y, long incy);
void add11(int n, int *x);
void muli22(int n, double alpha, double *x, double *y, int inc, int *ny);
void cblas_repro_dgbmv_n(long m, long n, long ku, long kl, double alpha, double *a, long lda,double *x, long incx, double *y, long incy);
double REPRO_SUM_K(long n, double *x, long inc_x);

int main(int argc,char** argv)
{
	if(argc != 13)
        {
                printf("dgemv <m> <n> <kl> <ku> <alpha> <lda> <incx> <beta> <incy> <epsilon> <type> <test_type>\n");
                exit(EXIT_FAILURE);
        }
	long m = atol(argv[1]);
	long n = atol(argv[2]);
	long kl = atol(argv[3]);
	long ku = atol(argv[4]);
	double alpha = atof(argv[5]);
	long lda = atol(argv[6]);
	long inc_x = atol(argv[7]);
	double beta = atof(argv[8]);
	long inc_y = atol(argv[9]);
	double epsilon = atof(argv[10]);
	long type = atol(argv[11]);
	char type2 = *argv[12];
	printf(" func=%s,m=%ld,n=%ld,kl=%ld,ku=%ld,alpha=%f,lda=%ld,inc_x=%ld,beta=%f,inc_y=%ld,epsilon=%e\n",argv[0],m,n,kl,ku,alpha,lda,inc_x,beta,inc_y,epsilon);
	printf("test openblas,openblas(re) and reproblas\n");
	long new_file_lines = (m>n?m:n)*(lda+inc_x+inc_y);
	long old_file_line;
	//long old_file_line = atol(getenv("OLD_FILE_LINES"));
	//printf("OLD_FILE_LINES=%ld\n",atol(getenv("OLD_FILE_LINES")));
	
	system("wc -l ./datefile");
	FILE *cmd = popen("wc -l ./datefile", "r");
	fscanf(cmd, "%ld %*s", &old_file_line);
	printf("OLD_FILE_LINES=%ld\n",old_file_line);

	printf("old_file_line VS new_file_lines = %ld %ld\n",old_file_line,new_file_lines);
	if(access("./datefile", F_OK) == -1 || new_file_lines > old_file_line){
		init_file(new_file_lines);
		char c[15];
		sprintf(c, "%ld", new_file_lines);
		setenv("OLD_FILE_LINES", c, 1);
		printf("c=%s, OLD_FILE_LINES=%ld\n",c,atol(getenv("OLD_FILE_LINES")));
	}
	switch(type){
		case 0:
			printf("CblasColMajor,CblasNoTrans\n");
			functional_test('c', 'n', m, n, kl, ku, alpha, lda, inc_x, beta, inc_y, epsilon);
			break;
		case 1:
			printf("CblasColMajor,CblasTrans\n");
			functional_test('c', 't', m, n, kl, ku, alpha, lda, inc_x, beta, inc_y, epsilon);
			break;
		case 2:
			printf("CblasRowMajor,CblasNoTrans\n");
			functional_test('r', 'n', m, n, kl, ku, alpha, lda, inc_x, beta, inc_y, epsilon);
			break;
		case 3:
			printf("CblasRowMajor,CblasTrans\n");
			functional_test('r', 't', m, n, kl, ku, alpha, lda, inc_x, beta, inc_y, epsilon);
			break;
	}
        return 0;
}
void functional_test(char order, char trans, long m, long n, long kl, long ku, double alpha, long lda, long inc_x, double beta, long inc_y, double epsilon){
	long dimx,dimy;

	if((order != 'c' && order != 'r') || (trans != 't' && trans != 'n')){
		printf("Error value of order/trans\n");
		exit(-1);
	}

	if(trans=='n'){
		dimx = 1 + (n - 1) * labs(inc_x);
	        dimy = 1 + (m - 1) * labs(inc_y);
	}else if(trans=='t'){
		dimx = 1 + (m - 1) * labs(inc_x);
                dimy = 1 + (n - 1) * labs(inc_y);
	}
	
	if(order=='c' && lda<m){
		printf("lda must be large than m.\n");
		exit(-1);
	}else if(order=='r' && lda<n){
		printf("lda must be large than n.\n");
                exit(-1);
	}

	long k = (order=='c')?n:m;
	long ldb = (kl+ku+1);
	double q[k];
	printf("q=%lx\n",q);
	printf("Start functional test.\n");
	double *x = (double*)malloc(sizeof(double) * dimx);
	double *y = (double*)malloc(sizeof(double) * dimy);
	double *y1 = (double*)malloc(sizeof(double) * dimy);
	double *a =  (double*)malloc(sizeof(double) * k*lda);
	double *b =  (double*)malloc(sizeof(double) * k*ldb);
	double *b0 =  (double*)malloc(sizeof(double) * k*lda);

	//double *y3 = (double*)malloc(sizeof(double) * dimy);
	//printf("A=%lx,B=%lx,B0=%lx,x=%lx,x0=%lx,y=%lx,y0=%lx,y1=%lx,y2=%lx\n",a,b,b0,x,x0,y,y0,y1,y2);
//	printf("a=%lx,b=%lx,b0=%lx,k=%ld y=%le\n",a,b,b0,k,y[dimy+2]);

	//init
	FILE *fp = fopen("./datefile","r");
        if(!fp){
                printf("Open File Error!\n");
	}
	printf("%lx\n",fp);
	int qn=0;
	unsigned long *qp = (unsigned long *)a;
	while(fscanf(fp, "%lx", &qp[qn++]) && qn<k*lda){
		if(feof(fp)) break;
        }
                printf("a,%ld\n",qn);

	qn=0;
	qp = (unsigned long *)x;
	while(fscanf(fp, "%lx", &qp[qn++]) && qn<dimx){
                if(feof(fp)) break;
        }
                printf("x,%ld\n",qn);

	qn=0;
	qp = (unsigned long *)y;
        while(fscanf(fp, "%lx", &qp[qn++]) && qn<dimy){
		if(feof(fp)) break;
	}
                printf("y,%ld\n",qn);
	
	printf("%lx\n",fp);
	printf("total lines=%ld\n",k*lda+dimx+dimy);
	fclose(fp);
	//srand(time(NULL));
	//init_mem(a,k*lda);
	//init_mem(x,dimx);
	//init_mem(y,dimy);
	if(order=='c'){
		ge_to_gb(m,n,ku,kl,ldb,lda,b,a);
		gb_to_ge(m,n,ku,kl,ldb,lda,b,b0);
	}else{
		ge_to_gb(n,m,kl,ku,ldb,lda,b,a);
		gb_to_ge(n,m,kl,ku,ldb,lda,b,b0);
	}
	int i;
//	for(i=0;i<k*lda;i++) printf("a[%d]=%le\tb0[%d]=%le\n",i,a[i],i,b0[i]);
//	for(i=0;i<k*ldb;i++) printf("b[%d]=%le\n",i,b[i]);
	for(i=0;i<dimx;i++) printf("x[%d]=%le\n",i,x[i]);


//	printf("\n----------openBLAS test----------\n");
//	memcpy(y2,y,dimy*sizeof(double));
//	memcpy(x0,x,dimx*sizeof(double));
//	memcpy(y0,y,dimy*sizeof(double));
        memcpy(y1,y,dimy*sizeof(double));
//	cblas_dgbmv(cbo(order),cbt(trans),m,n,kl,ku,alpha,b,ldb,x,inc_x,beta,y0,inc_y);
//	WRITEFILE("x86_64_cblas_dgbmv.log", (unsigned long *)y0, dimy);

	printf("\n----------openBLAS(re) test----------\n");
 //       memcpy(y2,y,dimy*sizeof(double));
//	WRITEFILE("b1.log",(unsigned long *)b,k*ldb);
//	WRITEFILE("x1.log",(unsigned long *)x,dimx);
	
	cblas_repro_dgbmv_d('c', 'n',m,n,kl,ku,alpha,b,ldb,x,inc_x,beta,y,inc_y);
//	WRITEFILE("b2.log",(unsigned long *)b,k*ldb);
//	WRITEFILE("x2.log",(unsigned long *)x,dimx);
	WRITEFILE("1111111111111.log", (unsigned long *)y, dimy);
	
	cblas_repro_dgbmv(CblasColMajor,CblasNoTrans,m,n,kl,ku,alpha,b,ldb,x,inc_x,beta,y1,inc_y);
//	WRITEFILE("b3.log",(unsigned long *)b,k*ldb);
//	WRITEFILE("x3.log",(unsigned long *)x,dimx);
	WRITEFILE("1111111111112.log", (unsigned long *)y1, dimy);

	check_double(y,y1,dimy,1,1,epsilon);
	free(a);
	free(b);
	free(x);
	free(y);
	//free(y3);
	printf("End functional test.\n");
}

void cblas_repro_dgbmv_d(char order, char trans,long m, long n, long ku, long kl, double alpha, double *b, long ldb,double *x, long incx, double beta, double *y, long incy){
	int i;
	for (i=0;i<m;i++)
		y[i]=beta*y[i];
	cblas_repro_dgbmv_n(m, n, kl, ku, alpha, b, ldb, x, incx, y, incy);

}

void add11(int n, int *x){
        int i;
        for(i=0;i<n;i++)
                x[i] += 1;
}
void muli22(int n, double alpha, double *x, double *y, int inc, int *ny){
        int i;
        for(i=0;i<n;i++)
                y[i*inc+ny[i]] = alpha * x[i];
}
void cblas_repro_dgbmv_n(long m, long n, long ku, long kl, double alpha,
          double *a, long lda,
          double *x, long incx, double *y, long incy){
        int i, offset_u, offset_l, start, end, length;
        offset_u = ku;
        offset_l = ku + m;

        int tld = kl+ku+1;
        double *tmp = (double *)malloc(m*tld*sizeof(double));
        int *ny = (int *)malloc(m*sizeof(int));
        memset(ny,0,m*sizeof(int));

        //y->tmp
        //COPY_K(m, y, incy, tmp, tld);
        //add1(m,ny);

        for (i = 0; i < MIN(n, m + ku); i++){
                start = MAX(offset_u, 0);
                end   = MIN(offset_l, ku + kl + 1);
                length  = end - start;

                //axpy(length, alpha*x[i], a+start, y+start-offset_u);
                muli22(length, x[i], a+start, tmp+(start-offset_u)*tld, tld, ny+start-offset_u);
                add11(length,ny+start-offset_u);


                offset_u --;
                offset_l --;
                a += lda;
        }
        for(i=0;i<m;i++)
	{
		//printf("REPRO_SUM_K %d | %lf\n",i,REPRO_SUM_K(ny[i], tmp+i*tld, 1));
                y[i*incy] += alpha*REPRO_SUM_K(ny[i], tmp+i*tld, 1);
	}
        free(tmp);
	free(ny);
}
double REPRO_SUM_K(long n, double *x, long inc_x){
 double  res = 0.0;
    double q1, M1, q2, M2;
    double max_num = fabs(x[0]);
    for(int i =1; i < n; i++){
        if(max_num < fabs(x[i])){
        max_num = fabs(x[i]);
        }
    }
    q1  = n*max_num/(1 - 2 * n * EPS_DOUBLE);
    M1 = pow(2, ceil(log2(q1)));
    q2 = n*( 2*EPS_DOUBLE *M1) / (1 - 2*n*EPS_DOUBLE);
    M2 = pow(2, ceil(log2(q2)));
    double res_high, res_low, temp_T_high, temp_T_low, temp_x_low;
    res_high = 0.0;
    res_low  = 0.0;
    for(int j = 0; j < n; j++){
        temp_T_high = M1 + x[j] - M1;
        temp_x_low = x[j] - temp_T_high;
        res_high = res_high + temp_T_high;
        temp_T_low = M2 + temp_x_low - M2;
        res_low = res_low + temp_T_low;
    }
    //printf("kernel N = %d\n",n);
    res = res_high + res_low;
    return res;
}
