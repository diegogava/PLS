#ifndef _PLS_H_
#define	_PLS_H_


#include <cublas.h>
#include <fstream>
#include <unordered_set>
#include <opencv2/opencv.hpp>

#include "util.h"

/*
 * PLS CPU
 */
class PLS_CPU {
public:
	
    void learn(const cv::Mat &_X, const cv::Mat &Y, const int factors,
            int max_factors = 1) {

        float avg = 0;
        float avg_last = 0;
        int n = 0;

        for (int nfactors = 1; nfactors <= max_factors; ++nfactors) {

            if (max_factors == 1) {
                nfactors = factors;
                max_factors = factors;
            }

            cv::Mat _Y;
            _Y = Y.clone();

            // preserve original matrix
            cv::Mat X = _X.clone();

            // calculate mean and std from X
            cv::Mat Xavg, Xstd;
            Xavg.create(1, X.cols, CV_32F);
            Xstd.create(1, X.cols, CV_32F);
            for (int col = 0; col < X.cols; ++col) {

                cv::Mat avg, std;
                cv::meanStdDev(X.col(col), avg, std);
                Xavg.ptr<float>(0)[col] = *avg.ptr<double>(0);
                Xstd.ptr<float>(0)[col] = *std.ptr<double>(0);

                // normalize X column
                for (int row = 0; row < X.rows; ++row) {
                    X.at<float>(row, col) -= Xavg.ptr<float>(0)[col];
                    X.at<float>(row, col) /= Xstd.ptr<float>(0)[col];
                }
            }

            // create labels
            std::vector<float> labels;
            labels.clear();
            {
                std::unordered_set<float> _labels;
                for (int i = 0; i < _Y.rows; ++i)
                    _labels.insert(_Y.ptr<float>(0)[i]);
                labels.reserve(_labels.size());
                for (float label : _labels)
                    labels.push_back(label);
            }

            // perform one-against-all
#ifdef __OPENMP_PLS__
#pragma omp parallel for schedule(guided)
#endif
            for (int bstar = 0; bstar < labels.size(); ++bstar) {

                counter_t counter;
                counter.tick();

                // reserve memory
                cv::Mat y;
                y.create(X.rows, 1, CV_32F);

                // set all target values to -1
                y = -1.0f;

                // set label target values to +1
                for (int i = 0; i < y.rows; ++i) {
                    if (_Y.ptr<float>(0)[i] == labels[bstar])
                        y.ptr<float>(0)[i] = 1.0f;
                }

                nipals(X.clone(), y, nfactors);
#ifdef __OPENMP_PLS__
#pragma omp critical
#endif
                {
                    std::cout << "PLS " << bstar << "/" << labels.size()
                            << " learned [" << counter.tock() << " seconds]" << std::endl;
                }
                
                // update ETR
                int seconds = counter.tock();
                n++;
                avg = avg + (seconds / (float) n) - (avg_last / n);
                avg_last = seconds;
                int rem_iters = (labels.size() - bstar - 1) + labels.size()*(max_factors - nfactors);
                long h = seconds/(60*60);
                long m = (seconds%(60*60))/60;
                long s = (seconds%(60*60))%60;
                long etr = avg * rem_iters;
                long etr_h = etr/(60*60);
                long etr_m = (etr%(60*60))/60;
                long etr_s = (etr%(60*60))%60;
                std::cout << "PLS learned [elapsed: " << h << ":" << m << ":" << s
                        << ", ETR: " << etr_h << ":" << etr_m << ":" << etr_s << "]" << std::endl;
            }
        }
        
        std::cout << "Average PLS learning time: " << avg << " seconds" << std::endl;
    }

private:

    cv::Mat nipals(cv::Mat X, cv::Mat Y, const int nfactors) {

        cv::Mat T, P, U, Q, W, B, _Y;

        // preserve Y
        _Y = Y.clone();

        //Setting the termination criteria
        int nMaxIterations, nMaxOuter = 1000;
        nMaxIterations = nfactors;
        double TermCrit = 10e-15, TempVal;
        cv::Mat tNorm;
        double MaxValX, MaxValY;
        int MaxIndexX, MaxIndexY;
        cv::Mat TempX, TempY;

        //Matrices for storing the intermediate values.
        cv::Mat tTemp, tNew, uTemp, wTemp, qTemp, pTemp, bTemp;

        //Allocating memory
        T.create(X.rows, nMaxIterations, CV_32F);
        P.create(X.cols, nMaxIterations, CV_32F);
        U.create(Y.rows, nMaxIterations, CV_32F);
        Q.create(Y.cols, nMaxIterations, CV_32F);
        W.create(X.cols, nMaxIterations, CV_32F);
        B.create(nMaxIterations, nMaxIterations, CV_32F);
        tTemp.create(X.rows, 1, CV_32F);
        uTemp.create(Y.rows, 1, CV_32F);

        for (int index1 = 0; index1 < nMaxIterations; index1++) {

            //Finding the column having the highest norm
            MaxValX = 0;
            MaxValY = 0;
            MaxIndexX = -10;
            MaxIndexY = -10;
            TempX.create(X.rows, 1, X.type());
            TempY.create(Y.rows, 1, Y.type());
            for (int index3 = 0; index3 < X.cols; index3++) {
                for (int index2 = 0; index2 < X.rows; index2++) {
                    TempX.at<float>(index2, 0) = X.at<float>(index2, index3);
                }
                if (cv::norm(TempX) > MaxValX) {
                    MaxValX = cv::norm(TempX);
                    MaxIndexX = index3;
                }
            }
            for (int index3 = 0; index3 < Y.cols; index3++) {
                for (int index2 = 0; index2 < Y.rows; index2++) {
                    TempY.at<float>(index2, 0) = Y.at<float>(index2, index3);
                }
                if (cv::norm(TempY) > MaxValY) {
                    MaxValY = cv::norm(TempY);
                    MaxIndexY = index3;
                }
            }

            for (int index3 = 0; index3 < X.rows; index3++) {
                tTemp.at<float>(index3, 0) = X.at<float>(index3, MaxIndexX);
                uTemp.at<float>(index3, 0) = Y.at<float>(index3, MaxIndexY);
            }

            // Iteration for Outer Modelling
            for (int index2 = 0; index2 < nMaxOuter; index2++) {

				std::cout << "Interação: " << index2 << "/" << nMaxOuter << std::endl;

                wTemp = X.t() * uTemp;
                wTemp = wTemp / cv::norm(wTemp);
                tNew = X * wTemp;
                qTemp = Y.t() * tNew;
                qTemp = qTemp / cv::norm(qTemp);
                uTemp = Y * qTemp;

                TempVal = cv::norm(tTemp - tNew);
                if (cv::norm(tTemp - tNew) < TermCrit) {
                    break;
                }
                tTemp = tNew.clone();
            }

            // Residual Deflation
            tNorm = tTemp.t() * tTemp;
            bTemp = uTemp.t() * tTemp / tNorm.at<float>(0, 0);
            pTemp = X.t() * tTemp / tNorm.at<float>(0, 0);
            X = X - tTemp * pTemp.t();
            Y = Y - bTemp.at<float>(0, 0) * (tTemp * qTemp.t());


            // Saving Results to Outputs.
            for (int index3 = 0; index3 != X.rows; index3++) {
                T.at<float>(index3, index1) = tTemp.at<float>(index3, 0);
                U.at<float>(index3, index1) = uTemp.at<float>(index3, 0);
            }
            for (int index3 = 0; index3 != X.cols; index3++) {
                P.at<float>(index3, index1) = pTemp.at<float>(index3, 0);
                W.at<float>(index3, index1) = wTemp.at<float>(index3, 0);
            }

            for (int index3 = 0; index3 != qTemp.rows; index3++) {
                Q.at<float>(index3, index1) = qTemp.at<float>(index3, 0);
            }
            B.at<float>(index1, index1) = bTemp.at<float>(0, 0);

            // Checking the residue
            if ((cv::norm(X) == 0) || (cv::norm(Y) == 0)) {
                break;
            }
        }
        // return BStar
        cv::Mat bstar = ((W * (P.t() * W).inv()) * (T.t() * T).inv() * T.t() * _Y);

        return bstar;
    }
};

/*
 * PLS GPU
 */
class PLS_GPU {
public:

    void learn(const cv::Mat &_X, const cv::Mat &Y, const int factors,
            int max_factors) {

        float avg = 0;
        float avg_last = 0;
        int n = 0;

        for (int nfactors = 1; nfactors <= max_factors; ++nfactors) {

            if (max_factors == 1) {
                nfactors = factors;
                max_factors = factors;
            }

            cv::Mat _Y;
            _Y = Y.clone();

            // preserve original matrix
            cv::Mat X = _X.clone();

            // create labels
            std::vector<float> labels;
            labels.clear();
            {
                std::unordered_set<float> _labels;
                for (int i = 0; i < _Y.rows; ++i)
                    _labels.insert(_Y.ptr<float>(0)[i]);
                labels.reserve(_labels.size());
                for (float label : _labels)
                    labels.push_back(label);
            }

            // perform one-against-all
            for (int bstar = 0; bstar < labels.size(); ++bstar) {

                counter_t counter;
                counter.tick();

                // reserve memory
                cv::Mat y;
                y.create(X.rows, 1, CV_32F);

                // set all target values to -1
                y = -1.0f;

                // set label target values to +1
                for (int i = 0; i < y.rows; ++i) {
                    if (_Y.ptr<float>(0)[i] == labels[bstar])
                        y.ptr<float>(0)[i] = 1.0f;
                }

                cv::Mat gpuX = X.clone();
                int gpuN = gpuX.rows;
                int gpud = gpuX.cols;
                cv::Mat gpuY = y.clone();
                int gpuf = gpuY.cols;
                cv::Mat gpuT, gpuP, gpuW, gpuB, gpuXmean, gpuXstd;
                gpuT.create(gpuN, nfactors, CV_32F);
                gpuP.create(gpud, nfactors, CV_32F);
                gpuW.create(gpud, nfactors, CV_32F);
                gpuXmean.create(1, gpud, CV_32F);
                gpuXstd.create(1, gpud, CV_32F);
                gpuB.create(1, nfactors, CV_32F);

                gpuNIPALS(&gpuX.at<float>(0), gpuN, gpud, &gpuY.at<float>(0),
                        gpuf, nfactors, &gpuT.at<float>(0), &gpuP.at<float>(0),
                        &gpuW.at<float>(0), &gpuB.at<float>(0),
                        &gpuXmean.at<float>(0), &gpuXstd.at<float>(0));

                // update ETR
                int seconds = counter.tock();
                n++;
                avg = avg + (seconds / (float) n) - (avg_last / n);
                avg_last = seconds;

                int rem_iters = (labels.size() - bstar - 1) + labels.size()*(max_factors - nfactors);
                long h = seconds/(60*60);
                long m = (seconds%(60*60))/60;
                long s = (seconds%(60*60))%60;
                long etr = avg * rem_iters;
                long etr_h = etr/(60*60);
                long etr_m = (etr%(60*60))/60;
                long etr_s = (etr%(60*60))%60;
                std::cout << "PLS learned [elapsed: " << h << ":" << m << ":" << s
                        << ", ETR: " << etr_h << ":" << etr_m << ":" << etr_s << "]" << std::endl;
            }
        }

        std::cout << "Average PLS learning time: " << avg << " seconds" << std::endl;
    }

private:

    void cublasSnormalize(int N, int d, float *d_X, float *mean, float *std) {

        float *temp = new float[N];
        for (int i = 0; i < N; i++)
            temp[i] = 1.0;

        float *d_mean, *d_ones;

        cublasAlloc(N, sizeof (d_ones[0]), (void**) &d_ones);
        cublasAlloc(d, sizeof (d_mean[0]), (void**) &d_mean);

        cublasSetVector(N, sizeof (temp[0]), temp, 1, d_ones, 1);
        cublasSgemv('T', N, d, (float) 1.0 / (float) N, d_X, N, d_ones, 1, 0.0, d_mean, 1);
        cublasGetVector(d, sizeof (temp[0]), d_mean, 1, mean, 1);

        cublasSgemm('N', 'T', N, d, 1, -1.0, d_ones, N, d_mean, d, 1.0, d_X, N);

        for (int i = 0; i < d; i++) {
            float s = (float) sqrt(cublasSdot(N, d_X + i*N, 1, d_X + i*N, 1) / (float) N);
            std[i] = s > 0.001 ? s : 1;
            cublasSscal(N, 1 / std[i], d_X + i*N, 1);
        }

        cublasFree(d_mean);
        cublasFree(d_ones);

        delete [] temp;
    }

    void gpuNIPALS(float *X, int N, int d, float *Y, int f, int numFactor, float *T, float *P, float *W, float *b, float *meanX, float *stdX) {

        cublasStatus status;
        status = cublasInit();
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Cublas initialization error\n");
            return;
        }

        float *d_X = 0;
        float *d_Y = 0;

        status = cublasAlloc(d*N, sizeof (X[0]), (void**) &d_X);
        status = cublasSetVector(N*d, sizeof (X[0]), X, 1, d_X, 1);
        status = cublasAlloc(f*N, sizeof (Y[0]), (void**) &d_Y);
        status = cublasSetVector(N*f, sizeof (Y[0]), Y, 1, d_Y, 1);

        cublasSnormalize(N, d, d_X, meanX, stdX);

        float *meanY = new float[f];
        float *stdY = new float[f];
        cublasSnormalize(N, f, d_Y, meanY, stdY);
        delete [] meanY;
        delete [] stdY;

        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Cublas error in X/Y\n");
            return;
        }

        float *d_t = 0;
        float *d_t0 = 0;
        float *d_w = 0;
        float *d_c = 0;
        float *d_u = 0;
        float *d_p = 0;

        status = cublasAlloc(N, sizeof (d_t[0]), (void**) &d_t);
        status = cublasAlloc(N, sizeof (d_t0[0]), (void**) &d_t0);
        status = cublasAlloc(d, sizeof (d_w[0]), (void**) &d_w);
        status = cublasAlloc(f, sizeof (d_c[0]), (void**) &d_c);
        status = cublasAlloc(N, sizeof (d_u[0]), (void**) &d_u);
        status = cublasAlloc(d, sizeof (d_p[0]), (void**) &d_p);

        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Cublas error in allocating memory to vectors\n");
            return;
        }

        for (int i = 0; i < numFactor; i++) {
            float eps = (float) 1e-6;
            cublasScopy(N, d_Y, 1, d_t, 1);
            cublasSscal(N, 1 / cublasSnrm2(N, d_t, 1), d_t, 1);
            cublasScopy(N, d_t, 1, d_u, 1);

            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("Cublas error in initializing iteration vectors\n");
                return;
            }
            for (int j = 0; j < 1; j++) {
                cublasScopy(N, d_t, 1, d_t0, 1);

                // w=normalize(X'*u)
                cublasSgemv('T', N, d, 1.0, d_X, N, d_u, 1, 0.0, d_w, 1);
                cublasSscal(d, 1 / cublasSnrm2(d, d_w, 1), d_w, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS) {
                    printf("Cublas error in X'*u\n");
                    return;
                }

                // t=normalize(X*w)
                cublasSgemv('N', N, d, 1.0, d_X, N, d_w, 1, 0.0, d_t, 1);
                cublasSscal(N, 1 / cublasSnrm2(N, d_t, 1), d_t, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS) {
                    printf("Cublas error in X*w\n");
                    return;
                }

                //c=normalize(Y'*t)            
                cublasSgemv('T', N, f, 1.0, d_Y, N, d_t, 1, 0.0, d_c, 1);
                cublasSscal(f, 1 / cublasSnrm2(f, d_c, 1), d_c, 1);
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS) {
                    printf("Cublas error in Y'*w\n");
                    return;
                }

                //u=(Y*c)
                cublasSgemv('N', N, f, 1.0, d_Y, N, d_c, 1, 0.0, d_u, 1);
                //cublasSscal(N,1/cublasSnrm2(N,d_u,1),d_u,1);	
                status = cublasGetError();
                if (status != CUBLAS_STATUS_SUCCESS) {
                    printf("Cublas error in Y*t\n");
                    return;
                }

                cublasSaxpy(N, -1.0, d_t, 1, d_t0, 1);
                float check = cublasSdot(N, d_t0, 1, d_t0, 1);
                if (check <= eps / 2.0)
                    break;

            }
            // p=(X'*t)
            cublasSgemv('T', N, d, 1.0, d_X, N, d_t, 1, 0.0, d_p, 1);
            b[i] = cublasSdot(N, d_u, 1, d_t, 1) / cublasSdot(N, d_t, 1, d_t, 1);

            // X&Y deflation
            cublasSgemm('N', 'T', N, d, 1, -1.0, d_t, N, d_p, d, 1.0, d_X, N);
            cublasSgemm('N', 'T', N, f, 1, (float) (-1.0 * b[i]), d_t, N, d_c, d, 1.0, d_Y, N);

            status = cublasGetError();
            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("Cublas error in X-Y deflation\n");
                return;
            }
            status = cublasGetVector(N, sizeof (d_t[0]), d_t, 1, T + i*N, 1);
            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("Cublas error in X-Y deflation\n");
                return;
            }

            status = cublasGetVector(d, sizeof (d_p[0]), d_p, 1, P + i*d, 1);
            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("Cublas error in X-Y deflation\n");
                return;
            }

            status = cublasGetVector(d, sizeof (d_w[0]), d_w, 1, W + i*d, 1);
            status = cublasGetError();
            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("Cublas error in X-Y deflation\n");
                return;
            }

            status = cublasGetError();
            if (status != CUBLAS_STATUS_SUCCESS) {
                printf("Cublas error in reading back GPU vector\n");
                return;
            }
        }

        status = cublasFree(d_X);
        status = cublasFree(d_Y);
        status = cublasFree(d_t);
        status = cublasFree(d_t0);
        status = cublasFree(d_w);
        status = cublasFree(d_c);
        status = cublasFree(d_u);
        status = cublasFree(d_p);
        status = cublasShutdown();
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error shutting down Cublas\n");
            return;
        }
    }
};

#endif
