// Solves the regularized inverse covariance matrix selection using a
// combination of Newton's method, quadratic approximation and
// coordinate descent.  The original algorithm was coded by Cho-Jui
// Hsieh.  Improvements were made by Matyas A. Sustik.
// This code is released under the GPL version 3.

// See the README file and QUIC.m for more information.
// Send questions, comments and license inquiries to: msustik@gmail.com

#define VERSION "1.2"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#ifdef GDEBUG
#include "startgdb.h"
#endif
#include "QUIC.h"
#ifdef LANG_R
  #include <R_ext/Print.h>
  #define MSG Rprintf
#endif
#if defined(LANG_M) || defined(MATLAB_MEX_FILE)
  #include <mex.h>
  #define MSG mexPrintf
#endif

#ifndef MSG
  #define MSG printf
#endif

#define EPS (double(2.22E-16))
//#define EPS ((double)0x3cb0000000000000)

#if !defined(LANG_R) && defined(_WIN32)
  #define dpotrf_ dpotrf
  #define dpotri_ dpotri
#endif

// It would be preferable to use an include such as lapack.h.  Except
// lapack.h is not available from the octave or liblapack-dev packages...
extern "C" {
    void dpotrf_(char* uplo, ptrdiff_t* n, double* A, ptrdiff_t* lda,
		 ptrdiff_t* info);
    void dpotri_(char* uplo, ptrdiff_t* n, double* A, ptrdiff_t* lda,
		 ptrdiff_t* info);
}

typedef struct {
    unsigned short i;
    unsigned short j;
} ushort_pair_t;

static inline unsigned long IsDiag(unsigned long p, const double* A)
{
    for (unsigned long k = 0, i = 0; i < p; i++, k += p)
	for (unsigned long j = 0; j < i; j++)
	    if (A[k + j] != 0.0)
		return 0;
    return 1;
}

static inline void CoordinateDescentUpdate(
    unsigned long p, const double* const S, const double* const Lambda,
    const double* X, const double* W, double* U, double* D,
    unsigned long i, unsigned long j, double& normD, double& diffD)
{
    unsigned long ip = i*p;
    unsigned long jp = j*p;
    unsigned long ij = ip + j;

    double a = W[ij]*W[ij];
    if (i != j)
        a += W[ip+i]*W[jp+j];

    double b = S[ij] - W[ij];
    for (unsigned long k = 0; k < p ; k++)
        b += W[ip+k]*U[k*p+j];

    double l = Lambda[ij]/a;
    double c = X[ij] + D[ij];
    double f = b/a;
    double mu;
    normD -= fabs(D[ij]);
    if (c > f) {
        mu = -f - l;
        if (c + mu < 0.0) {
            mu = -c;
	    D[ij] = -X[ij];
	} else {
	    D[ij] += mu;
	}
    } else {
	mu = -f + l;
	if (c + mu > 0.0) {
	    mu = -c;
	    D[ij] = -X[ij];
	} else {
	    D[ij] += mu;
	}
    }
    diffD += fabs(mu);
    normD += fabs(D[ij]);
    if (mu != 0.0) {
        for (unsigned long k = 0; k < p; k++)
            U[ip+k] += mu*W[jp+k];
        if (i != j) {
            for (unsigned long k = 0; k < p; k++)
                U[jp+k] += mu*W[ip+k];
        }
    }
}

// Return the objective value.
static inline double DiagNewton(unsigned long p, const double* S,
				const double* Lambda, const double* X,
				const double* W, double* D)
{
    for (unsigned long ip = 0, i = 0; i < p; i++, ip += p) {
	for (unsigned long jp = 0, j = 0; j < i; j++, jp += p) {
	    unsigned long ij = ip + j;
	    double a = W[ip + i]*W[jp + j];
	    double b = S[ij];
	    double l = Lambda[ij]/a;
	    double f = b/a;
	    double mu;
	    if (0 > f) {
		mu = -f - l;
		if (mu < 0.0) {
		    mu = 0.0;
		    D[ij] = -X[ij];
		} else {
		    D[ij] += mu;
		}
	    } else {
		mu = -f + l;
		if (mu > 0.0) {
		    mu = 0.0;
		    D[ij] = -X[ij];
		} else {
		    D[ij] += mu;
		}
	    }
	}
    }
    double logdet = 0.0;
    double l1normX = 0.0;
    double trSX = 0.0;
    for (unsigned long i = 0, k = 0; i < p; i++, k += (p+1)) {
	logdet += log(X[k]);
	l1normX += fabs(X[k])*Lambda[k];
	trSX += X[k]*S[k];
	double a = W[k]*W[k];
	double b = S[k] - W[k];
	double l = Lambda[k]/a;
	double c = X[k];
	double f = b/a;
	double mu;
	if (c > f) {
	    mu = -f - l;
	    if (c + mu < 0.0) {
		D[k] = -X[k];
		continue;
	    }
	} else {
	    mu = -f + l;
	    if (c + mu > 0.0) {
		D[k] = -X[k];
		continue;
	    }
	}
	D[k] += mu;
    }
    double fX = -logdet + trSX + l1normX;
    return fX;
}

static double projLogDet(unsigned long p, const double* S,
			 double* W, double* prW, double* Lambda)
{
    // The computed W does not satisfy |W - S| .< Lambda.  Project it.
    for (unsigned long i = 0; i < p; i++) {
	for (unsigned long j = 0; j <= i; j++) {
	    double tmp = W[i*p+j];
	    if (S[i*p+j] - Lambda[i*p+j] > tmp)
		tmp = S[i*p+j] - Lambda[i*p+j];
	    if (S[i*p+j] + Lambda[i*p+j] < tmp)
		tmp = S[i*p+j] + Lambda[i*p+j];
	    prW[i*p+j] = tmp;
	    prW[j*p+i] = tmp;
	}
    }
    ptrdiff_t info = 0;
    ptrdiff_t p0 = p;
    dpotrf_((char*) "U", &p0, prW, &p0, &info);
    if (info != 0)
	return 1e+15;
    double logdet = 0.0;
    for (unsigned long i = 0, k = 0; i < p; i++, k += (p+1))
	logdet += log(prW[k]);
    logdet += logdet;
    return logdet;
}

#define QUIC_MSG_NO      0
#define QUIC_MSG_MIN     1
#define QUIC_MSG_NEWTON  2
#define QUIC_MSG_CD      3
#define QUIC_MSG_LINE    4

// mode = {'D', 'P', 'T'} for 'default', 'path' or 'trace'.
//extern "C"
void QUIC(char mode, uint32_t& p, const double* S, double* Lambda0,
	  uint32_t& pathLen, const double* path, double& tol,
	  int32_t& msg, uint32_t& maxIter,
	  double* X, double* W, double* opt, double* cputime,
	  uint32_t* iter, double* dGap)
{
#ifdef GDEBUG
    startgdb();
#endif
    if (mode >= 'a')
	mode -= ('a' - 'A');
    if (msg >= QUIC_MSG_MIN) {
	const char* modes[] = { ".", " in path mode.", " in trace mode." };
	const char* mode_str = modes[0];
	if (mode == 'P')
	    mode_str = modes[1];
	else if (mode == 'T')
	    mode_str = modes[2];;
	MSG("Running QUIC v%s%s\n", VERSION, mode_str);
    }


    double timeBegin = clock();
    srand(1);
    unsigned long maxNewtonIter = maxIter;
    double cdSweepTol = 0.05;
    unsigned long max_lineiter = 20;
    double fX = 1e+15;
    double fX1 = 1e+15;
    double fXprev = 1e+15;
    double sigma = 0.001;
    double* D = (double*) malloc(p*p*sizeof(double));
    double* U = (double*) malloc(p*p*sizeof(double));
    double* Lambda;
    if (pathLen > 1) {
	Lambda = (double*) malloc(p*p*sizeof(double));
	for (unsigned long i = 0; i < p*p; i++)
	    Lambda[i] = Lambda0[i]*path[0];
    } else {
	Lambda = Lambda0;
    }
    ushort_pair_t* activeSet = (ushort_pair_t*)
	malloc(p*(p+1)/2*sizeof(ushort_pair_t));
    double l1normX = 0.0;
    double trSX = 0.0;
    double logdetX = 0.0;
    for (unsigned long i = 0, k = 0; i < p ; i++, k += p) {
	for (unsigned long j = 0; j < i; j++) {
	    l1normX += Lambda[k+j]*fabs(X[k+j]);
	    trSX += X[k+j]*S[k+j];
	}
    }
    l1normX *= 2.0;
    trSX *= 2.0;
    for (unsigned long i = 0, k = 0; i < p; i++, k += (p+1)) {
	l1normX += Lambda[k]*fabs(X[k]);
	trSX += X[k]*S[k];
    }
    unsigned long pathIdx = 0;
    unsigned long NewtonIter = 1;
    for (; NewtonIter <= maxNewtonIter; NewtonIter++) {
	double normD = 0.0;
	double diffD = 0.0;
	double subgrad = 1e+15;
	if (NewtonIter == 1 && IsDiag(p, X)) {
	    if (msg >= QUIC_MSG_NEWTON) {
		MSG("Newton iteration 1.\n");
		MSG("  X is a diagonal matrix.\n");
	    }
	    memset(D, 0, p*p*sizeof(double));
	    fX = DiagNewton(p, S, Lambda, X, W, D);
	} else {
	    // Compute the active set and the minimum norm subgradient:
	    unsigned long numActive = 0;
	    memset(U, 0, p*p*sizeof(double));
	    memset(D, 0, p*p*sizeof(double));
	    subgrad = 0.0;
	    for (unsigned long k = 0, i = 0; i < p; i++, k += p) {
		for (unsigned long j = 0; j <= i; j++) {
		    double g = S[k+j] - W[k+j];
		    if (X[k+j] != 0.0 || (fabs(g) > Lambda[k+j])) {
			activeSet[numActive].i = (unsigned short) i;
			activeSet[numActive].j = (unsigned short) j;
			numActive++;
			if (X[k+j] > 0)
			    g += Lambda[k+j];
			else if (X[k+j] < 0)
			    g -= Lambda[k+j];
			else
			    g = fabs(g) - Lambda[k+j];
			subgrad += fabs(g);
		    }
		}
	    }
	    if (msg >= QUIC_MSG_NEWTON) {
		MSG("Newton iteration %ld.\n", NewtonIter);
		MSG("  Active set size = %ld.\n", numActive);
		MSG("  sub-gradient = %e, l1-norm of X = %e.\n",
		       subgrad, l1normX);
	    }
	    for (unsigned long cdSweep = 1; cdSweep <= 1 + NewtonIter/3;
		 cdSweep++) {
		diffD = 0.0;

		for (unsigned long i = 0; i < numActive; i++ ) {
		    unsigned long j = i + rand()%(numActive - i);
		    unsigned short k1 = activeSet[i].i;
		    unsigned short k2 = activeSet[i].j;
		    activeSet[i].i = activeSet[j].i;
		    activeSet[i].j = activeSet[j].j;
		    activeSet[j].i = k1;
		    activeSet[j].j = k2;
		}
		for (unsigned long l = 0; l < numActive; l++) {
		    unsigned long i = activeSet[l].i;
		    unsigned long j = activeSet[l].j;
		    CoordinateDescentUpdate(p, S, Lambda, X, W,
					    U, D, i, j, normD,
					    diffD);
		}
		if (msg >= QUIC_MSG_CD) {
		    MSG("  Coordinate descent sweep %ld. norm of D = %e, "
			   "change in D = %e.\n", cdSweep, normD, diffD);
		}
		if (diffD <= normD*cdSweepTol)
		    break;
	    }
	}
	if (fX == 1e+15) {
	    // Note that the upper triangular part is the lower
	    // triangular part for the C arrays.
	    ptrdiff_t info = 0;
	    ptrdiff_t p0 = p;
	    memcpy(U, X, sizeof(double)*p*p);
	    dpotrf_((char*) "U", &p0, U, &p0, &info);
	    if (info != 0) {
		MSG("Error! Lack of positive definiteness!");
		iter[0] = -1;
		free(activeSet);
		free(U);
		free(D);
		if (pathLen > 1)
		    free(Lambda);
		return;
	    }
	    for (unsigned long i = 0, k = 0; i < p; i++, k += (p+1))
		logdetX += log(U[k]);
	    logdetX *= 2.0;
	    fX = (trSX + l1normX) - logdetX;
	}
	double trgradgD = 0.0;
	for (unsigned long i = 0, k = 0; i < p ; i++, k += p)
	    for (unsigned long j = 0; j < i; j++)
		trgradgD += (S[k+j]-W[k+j])*D[k+j];
	trgradgD *= 2.0;
	for (unsigned long i = 0, k = 0; i < p; i++, k += (p+1))
	    trgradgD += (S[k]-W[k])*D[k];

	double alpha = 1.0;
	double l1normXD = 0.0;
	double fX1prev = 1e+15;
	for (unsigned long lineiter = 0; lineiter < max_lineiter;
	     lineiter++) {
	    double l1normX1 = 0.0;
	    double trSX1 = 0.0;
	    for (unsigned long i = 0, k = 0; i < p ; i++, k += p) {
		for (unsigned long j = 0; j < i; j++) {
		    unsigned long ij = k + j;
		    W[ij] = X[ij] + D[ij]*alpha;
		    l1normX1 += fabs(W[ij])*Lambda[ij];
		    trSX1 += W[ij]*S[ij];
		}
	    }
	    l1normX1 *= 2.0;
	    trSX1 *= 2.0;
	    for (unsigned long i = 0, k = 0; i < p; i++, k += (p+1)) {
		W[k] = D[k]*alpha + X[k];
		l1normX1 += fabs(W[k])*Lambda[k];
		trSX1 += W[k]*S[k];
	    }
	    // Note that the upper triangular part is the lower
	    // triangular part for the C arrays.
	    ptrdiff_t info = 0;
	    ptrdiff_t p0 = p;
	    // We are using W to hold the Cholesky decomposition; W
	    // will hold the inverse later on.
	    dpotrf_((char*) "U", &p0, W, &p0, &info);
	    if (info != 0) {
		if (msg >= QUIC_MSG_LINE)
		    MSG("    Line search step size %e.  Lack of positive "
			   "definiteness.\n", alpha);
		alpha *= 0.5;
		continue;
	    }
	    double logdetX1 = 0.0;
	    for (unsigned long i = 0, k = 0; i < p; i++, k += (p+1))
		logdetX1 += log(W[k]);
	    logdetX1 *= 2.0;
	    fX1 = (trSX1 + l1normX1) - logdetX1;
	    if (alpha == 1.0)
		l1normXD = l1normX1;
	    if (fX1 <= fX + alpha*sigma*(trgradgD + l1normXD - l1normX) ||
		normD == 0) {
		if (msg >= QUIC_MSG_LINE)
		    MSG("    Line search step size chosen: %e.\n", alpha);
		fXprev = fX;
		fX = fX1;
		l1normX = l1normX1;
		logdetX = logdetX1;
		trSX = trSX1;
		break;
	    }
	    if (msg >= QUIC_MSG_LINE) {
		MSG("    Line search step size %e.\n", alpha);
		MSG("      Objective value would not decrease sufficiently: "
		    "%e.\n", fX1 - fX);
	    }
	    if (fX1prev < fX1) {
		fXprev = fX;
		l1normX = l1normX1;
		logdetX = logdetX1;
		trSX = trSX1;
		break;
	    }
	    fX1prev = fX1;
	    alpha *= 0.5;
	}
	if (msg >= QUIC_MSG_NEWTON) {
	    MSG("  Objective value decreased by %e.\n", fXprev - fX);
	}
	// compute W = inv(X):
	ptrdiff_t info;
	ptrdiff_t p0 = p;
	dpotri_((char*) "U", &p0, W, &p0, &info);

	for (unsigned long i = 0; i < p; i++) {
	    for (unsigned long j = 0; j <= i; j++) {
		double tmp = W[i*p+j];
		W[j*p+i] = tmp;
	    }
	}
	for (unsigned long i = 0, k = 0; i < p; i++, k += p)
	    for (unsigned long j = 0; j <= i; j++)
		X[k+j] += alpha*D[k+j];
	if (mode == 'T') {
	    if (opt != NULL)
		opt[NewtonIter - 1] = fX;
	    if (cputime != NULL) {
		// Note that nn trace mode we do not want to count the
		// time spent computing the duality gap.
		cputime[NewtonIter - 1] = (clock()-timeBegin)/CLOCKS_PER_SEC;
	    }
	    if (dGap != NULL) {
		double logdetW = projLogDet(p, S, W, U, Lambda);
		double gap = -logdetW - p - logdetX + trSX + l1normX;
		dGap[NewtonIter - 1] = gap;
	    }
	}
	// Check for convergence.
	if (subgrad*alpha >= l1normX*tol && (fabs((fX - fXprev)/fX) >= EPS))
	    continue;
	if (mode =='P') {
	    if (opt != NULL)
		opt[pathIdx] = fX;
	    if (iter != NULL)
		iter[pathIdx] = NewtonIter;
	    if (dGap != NULL) {
		double logdetW = projLogDet(p, S, W, U, Lambda);
		double gap = -logdetW - p - logdetX + trSX + l1normX;
		dGap[pathIdx] = gap;
	    }
	    for (unsigned long i = 0, k = 0; i < p; i++, k += p)
		for (unsigned long j = i+1; j < p; j++)
		    X[k+j] = X[j*p+i];
	    double elapsedTime = (clock() - timeBegin)/CLOCKS_PER_SEC;
	    if (cputime != NULL)
		cputime[pathIdx] = elapsedTime;
	    // Next lambda.
	    pathIdx++;
	    if (pathIdx == pathLen)
		break;
	    if (msg > QUIC_MSG_NO)
		MSG("  New scaling value: %e\n", path[pathIdx]);
	    unsigned long p2 = p*p;
	    memcpy(X + p2, X, p2*sizeof(double));
	    memcpy(W + p2, W, p2*sizeof(double));
	    X += p2;
	    W += p2;
	    for (unsigned long i = 0; i < p*p; i++)
		Lambda[i] = Lambda0[i]*path[pathIdx];
	    l1normX = l1normX/path[pathIdx-1]*path[pathIdx];
	    continue;
	}
	break;
    }
    if (mode == 'D') {
	if (opt)
	    opt[0] = fX;
	if (dGap != NULL) {
	    double logdetW = projLogDet(p, S, W, U, Lambda);
	    double gap = -logdetW - p - logdetX + trSX + l1normX;
	    dGap[0] = gap;
	}
	if (iter != NULL)
	    iter[0] = NewtonIter;
    }
    if (mode == 'T' && iter != NULL)
	iter[0] = NewtonIter - 1;
    free(activeSet);
    free(U);
    free(D);
    if (pathLen > 1)
	free(Lambda);
    for (unsigned long i = 0, k = 0; i < p; i++, k += p)
	for (unsigned long j = i+1; j < p; j++)
	    X[k+j] = X[j*p+i];
    double elapsedTime = (clock() - timeBegin)/CLOCKS_PER_SEC;
    if (mode == 'D' && cputime != NULL)
	cputime[0] = elapsedTime;
    if (msg >= QUIC_MSG_MIN)
	MSG("QUIC CPU time: %.3f seconds\n", elapsedTime);
}

// extern "C"
// void QUICR(char** modePtr, uint32_t& p, const double* S, double* Lambda0,
// 	   uint32_t& pathLen, const double* path, double& tol,
// 	   int32_t& msg, uint32_t& maxIter,
// 	   double* X, double* W, double* opt, double* cputime,
// 	   uint32_t* iter, double* dGap)
// {
//     char mode = **modePtr;
//     QUIC(mode, p, S, Lambda0, pathLen, path, tol, msg, maxIter, X, W,
// 	 opt, cputime, iter, dGap);
// }
