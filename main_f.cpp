#include <stdio.h>
#include <iostream>
#include<stdlib.h>
#include<math.h>
#include "omp.h"
#include<string.h>
#include <chrono>
#include <time.h>
#include <random>
#include <cmath>
#include <iomanip>
#include <bits/stdc++.h>


# define N_PATHS 1000
# define DAYS 365.25
# define HOURS 24
# define MINUTES 60
# define SECONDS 60
# define MAX_DIMS 10


void cholesky_matrix (double rho_matrix[][MAX_DIMS], double cho_matrix[][MAX_DIMS], std::uint16_t n_size)
{
    double lower[n_size][n_size] {0.};

    for (int i = 0; i < n_size; i++) {
        for (int j = 0; j <= i; j++) {
            double sum = 0;
            for (int k = 0; k < j; k++) {
                sum += lower[i][k] * lower[j][k];
            }
            if (i == j) {
                lower[i][j] = sqrt(rho_matrix[i][i] - sum);
            }
            else {
                lower[i][j] = (1.0 / lower[j][j] * (rho_matrix[i][j] - sum));
            }
        }
    }
 
    // // Displaying Lower Triangular and its Transpose
    // std::cout << " Lower Triangular" << "Transpose" << std::endl;
    // for (int i = 0; i < n_size; i++) {
         
    //     // Lower Triangular
    //     for (int j = 0; j < n_size; j++)
    //         std::cout << lower[i][j] << "\t";
    //     std::cout << "\t";
         
    //     // Transpose of Lower Triangular
    //     for (int j = 0; j < n_size; j++)
    //         std::cout << lower[j][i] << "\t";
    //     std::cout << std::endl;
    // }

    //double t_lower [n_size][n_size];
    // Transpose Lower Triangular
    for (int i = 0; i < n_size; i++) {
        for (int j = 0; j < n_size; j++)
            cho_matrix[i][j] = lower[j][i];
    }
}

double start_exp (
    std::uint16_t n_assets,
    double T, 
    double discount_f,
    double rho_ij,
    double K,
    double F,
    double sigma,
    double wi,
    int paths,
    int path_len
)
{

    double sigma2 = sigma * sigma / 2;
    double dt = T / path_len;
    double sqrt_dt = sqrt(dt);

    double rho_matrix[n_assets][MAX_DIMS];
    for (int i = 0; i < n_assets; ++i){
        for (int j = 0; j < n_assets; ++j){
            rho_matrix[i][j] = rho_ij;
        }
        rho_matrix[i][i] = 1; 
    }

    // for (int i = 0; i < n_assets; ++i){
    //     for (int j = 0; j < n_assets; ++j){
    //         std::cout << rho_matrix[i][j] << ' ';
    //     };
    //     std::cout << std::endl;
    // }

    double t_lower [n_assets][MAX_DIMS];
    cholesky_matrix(rho_matrix, t_lower, n_assets);

    // std::cout << "now we get Cholesky decomposition" << std::endl;
    // for (int i = 0; i < n_assets; ++i){
    //     for (int j = 0; j < n_assets; ++j){
    //         std::cout << t_lower[i][j] << ' ';
    //     }
    //     std::cout << std::endl;
    // }

    //omp_set_num_threads(8);
    std::uint64_t path_num{};
    double payoff = 0;

    //#pragma omp parallel for private(path_num)
    for (path_num = 0; path_num < paths; ++path_num)
    {
        // auto millisec_since_epoch = std::chrono::duration_cast<
        //     std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::normal_distribution<double> d{0.0, 1.0};
        auto dW = [&d, &gen]{ return d(gen); };
        double normal_vector[n_assets] {0.};
        double W_path[n_assets] {0.};

        for (int p = 0; p < path_len; ++p)
        {
            for (int i = 0; i < n_assets; ++i){
                normal_vector[i] = dW() * sigma * sqrt_dt;
            }
            for (int i = 0; i < n_assets; ++i){
                for (int j = 0; j < n_assets; ++j){
                    W_path[i] += normal_vector[j] * t_lower[j][i];
                }
            }
        }
        double S_paths[n_assets] {0.};
        double B = 0;
        for (int i = 0; i < n_assets; ++i){
            S_paths[i] = F * exp(-T * sigma2 + W_path[i]);
            B += S_paths[i];
        }
        B = B / n_assets;
        double cur_payoff = std::max(B - K, 0.);
        payoff += cur_payoff;
    }
    payoff /= paths;
    //std::cout << "Average payoff = " << payoff << std::endl;
    return payoff;

}

int main(int argc, char *argv[])
{
    std::uint16_t n_assets = 4;

    double T = 5;
    double discount_f = 1;
    double rho_ij = 0.5; // [6] = {0.1, 0.3, 0.5, 0.7, 0.8, 0.95};
    double K = 100;
    double F [11] = {50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150}; // = 100;
    double sigma = 0.4;
    
    double wi = 1 / n_assets;
    double avg_payoff = 0.;

    int paths = 10'000;
    int path_len = 1000;

    std::uint8_t loop_num = 100;

    float global_elapsed_seconds;
    auto global_start = std::chrono::system_clock::now();

    for (int r = 0; r < 11; r++)
    {
        float inner_seconds;
        float seconds_record[1] {0.};

        std::ofstream myfile;
        std::string name = "time_control_f_" + std::to_string(F[r]) + "_single_core.csv";
        myfile.open (name);

        for (int l = 0; l < loop_num; ++l)
        {        
            auto inner_start = std::chrono::system_clock::now();

            avg_payoff = start_exp(
                n_assets, T, discount_f, rho_ij, K, F[r], sigma, wi, paths, path_len
            );

            auto inner_end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = inner_end - inner_start;
            seconds_record[0] = elapsed_seconds.count();

            myfile << avg_payoff << ';' << seconds_record[0] << "\n";
            std::cout << "Inner elapsed = " << seconds_record[0] << std::endl;
        }
        myfile.close();
    }

    auto global_end = std::chrono::system_clock::now();
    std::chrono::duration<double> glob_elapsed_seconds = global_end - global_start;
    printf("Global time elapsed = %f", glob_elapsed_seconds);

    return 0;
}
