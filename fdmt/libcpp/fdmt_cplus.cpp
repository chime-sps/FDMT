// g++ -Wall -O2 -fopenmp -shared -Wl,-soname,fdmt_cplus -o fdmt_cplus.so -fPIC fdmt_cplus.cpp

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace std;

extern "C" 
void fdmt_iter_par(double* fs,
                   int nchan,
                   float df,
                   int32_t* Q,
                   float* src,
                   float* dest,
                   int i,
                   float fmin,
                   float fmax,
                   int maxDT) {
    int T;
    float dF;
    double* f_starts;
    double* f_ends;
    double* f_mids;
    int i_F;
    float f0, f1, f2, cor, C, C01, C12, loc, glo;
    int R, i_dT, dT_mid01, dT_mid12, dT_rest;

    T = int(src[nchan]);
    dF = df * pow(2, i);
    f_starts = fs;
    f_ends = new double[nchan];
    f_mids = new double[nchan];
    int i_max = nchan / int(pow(2, i));

    for (i_F = 0; i_F < i_max; ++i_F) {
        f0 = f_starts[i_F];
        f1 = f_mids[i_F];
        f2 = f_ends[i_F];
        cor = (i > 1) ? df : 0;

        C = (pow(f1, -2) - pow(f0, -2)) / (pow(f2, -2) - pow(f0, -2));
        C01 = ((pow(f1 - cor, -2) - pow(f0, -2)) / (pow(f2, -2) - pow(f0, -2)));
        C12 = ((pow(f1 + cor, -2) - pow(f0, -2)) / (pow(f2, -2) - pow(f0, -2)));

        loc = pow(f0, -2) - pow((f0 + dF), -2);
        glo = pow(fmin, -2) - pow(fmax, -2);
        R = (maxDT - 1) * loc / glo + 2;

        for (i_dT = 0; i_dT < R; ++i_dT) {
            dT_mid01 = round(i_dT * C01);
            dT_mid12 = round(i_dT * C12);
            dT_rest = i_dT - dT_mid12;
            int index1 = Q[i * nchan + i_F] + i_dT;
            int index2 = Q[(i - 1) * nchan + 2 * i_F] + dT_mid01;
            int index3 = Q[(i - 1) * nchan + 2 * i_F + 1] + dT_rest;
            for (int k = 0; k < T; ++k) {
                dest[index1 * T + k] = src[index2 * T + k];
                for (int l = dT_mid12; l < T; ++l) {
                    dest[index1 * T + k] += src[index3 * T + l - dT_mid12];
                }
            }
        }
    }
}

extern "C" 
void buildA(float* A, float* B, int* Q, float* spectra, long long* DTplan, int num_rows_A, int num_cols_A, int num_elements_Q, int num_time_steps) {
    cout << "num_time_steps " << num_time_steps << endl;
    cout << "num_elements_Q " << num_elements_Q << endl;
    cout << "num_rows_A " << num_rows_A << endl;
    cout << "num_cols_A " << num_cols_A << endl;
    cout << "DTplan[0] " << DTplan[0] << endl;
    for (int t = 1; t <= num_time_steps; ++t) {
        int idx = Q[t - 1];
        for (int i = 0; i < num_elements_Q; ++i) {
            for (int j = i; j < num_rows_A; ++j) {
                A[(idx + i) * num_cols_A + j] = A[(idx + i - 1) * num_cols_A + j] + spectra[t * num_cols_A + j - i];
            }
        }
    }
    for (int t = 1; t <= num_time_steps; ++t) {
        int idx = Q[t - 1];
        for (int i = 0; i < num_elements_Q; ++i) {
            for (int j = i; j < num_rows_A; ++j) {
                A[(idx + i) * num_cols_A + j] /= (i + 1);
            }
        }
    }
}
