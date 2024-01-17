/***************************************************************************
Copyright (c) 2013, The OpenBLAS Project
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the
distribution.
3. Neither the name of the OpenBLAS project nor the names of
its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE OPENBLAS PROJECT OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*****************************************************************************/


/**************************************************************************************
 * * 2013/09/14 Saar
 * *     BLASTEST float         : OK
 * *     BLASTEST double        : OK
 *       CTEST                  : OK
 *       TEST                   : OK
 * *
 * **************************************************************************************/

#include "common.h"

void add1(int n, int *x){
	int i;
	for(i=0;i<n;i++)
		x[i] += 1;
}
void muli2(int n, double alpha, double *x, double *y, int inc, int *ny){
	int i;
        for(i=0;i<n;i++)
                y[i*inc+ny[i]] = alpha * x[i];
}

void CNAME(BLASLONG m, BLASLONG n, BLASLONG ku, BLASLONG kl, FLOAT alpha,
          FLOAT *a, BLASLONG lda,
          FLOAT *x, BLASLONG incx, FLOAT *y, BLASLONG incy, void *buffer){
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
		muli2(length, x[i], a+start, tmp+(start-offset_u)*tld, tld, ny+start-offset_u);
		add1(length,ny+start-offset_u);

		//trans?
//		y[i*incy] = alpha * ACCU_ReproDOT(length, a + start, 1, x+(start - offset_u)*incx, incx) + beta*y[i*incy];
//		y[i*incy] = alpha * dot(length, a + start, 1, x+(start - offset_u)*incx, incx) + beta*y[i*incy];

		offset_u --;
		offset_l --;
		a += lda;
	}
	for(i=0;i<m;i++)
		y[i*incy] += alpha*REPRO_SUM_K(ny[i], tmp+i*tld, 1);
	free(tmp);
	free(ny);
}
