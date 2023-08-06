// Markus Buchholz, 2023
// g++ x_cpg3.cpp -o t -I/usr/include/python3.8 -lpython3.8

#include <iostream>
#include <vector>
#include <random>
#include <math.h>
#include <cmath>

#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

float dt = 0.01f;
float freq_a = 0.2f;
float freq_b = 0.18f;
float freq_c = 0.16f;

float coupling_strength_ab = 0.3f;
float coupling_strength_bc = 0.2f;
float coupling_strength_ca = 0.1f;

//------------------------------------------------------------------------------------
float generateRandom(float a, float b)
{

    std::random_device engine;
    std::mt19937 gen(engine());
    std::uniform_real_distribution<float> distribution(a, b * M_PI);

    return distribution(gen);
}

//------------------------------------------------------------------------------------
// dtheta_a
float function1(float theta_a, float theta_b, float theta_c)
{

    return 2 * M_PI * freq_a + coupling_strength_ab * std::sin(theta_b - theta_a) + coupling_strength_ca * std::sin(theta_c - theta_a);
}

//------------------------------------------------------------------------------------
// dtheta_b
float function2(float theta_a, float theta_b, float theta_c)
{

    return 2 * M_PI * freq_b + coupling_strength_bc * std::sin(theta_c - theta_b) + coupling_strength_ab * std::sin(theta_a - theta_b);
}

//------------------------------------------------------------------------------------
// dtheta_c
float function3(float theta_a, float theta_b, float theta_c)
{

    return 2 * M_PI * freq_c + coupling_strength_ca * std::sin(theta_a - theta_c) + coupling_strength_bc * std::sin(theta_b - theta_c);
}

//------------------------------------------------------------------------------------
std::tuple<std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>> methodRungeKutta1Diff()
{

    std::vector<float> diffEq1;
    std::vector<float> diffEq2;
    std::vector<float> diffEq3;

    std::vector<float> time;

    // init values
    float x1 = generateRandom(0.0f, 0.1f);  // x1
    float x2 = generateRandom(1.1f, 1.11f); // x2
    float x3 = generateRandom(1.9f, 2.0f);  // x3
    float t = 0.0;                          // init time

    diffEq1.push_back(x1);
    diffEq2.push_back(x2);
    diffEq3.push_back(x3);
    time.push_back(t);

    for (int ii = 0; ii < 2000; ii++)
    {
        t = t + dt;
        float k11 = function1(x1, x2, x3);
        float k12 = function2(x1, x2, x3);
        float k13 = function3(x1, x2, x3);

        float k21 = function1(x1 + dt / 2 * k11, x2 + dt / 2 * k12, x3 + dt / 2 * k13);
        float k22 = function2(x1 + dt / 2 * k11, x2 + dt / 2 * k12, x3 + dt / 2 * k13);
        float k23 = function3(x1 + dt / 2 * k11, x2 + dt / 2 * k12, x3 + dt / 2 * k13);

        float k31 = function1(x1 + dt / 2 * k21, x2 + dt / 2 * k22, x3 + dt / 2 * k23);
        float k32 = function2(x1 + dt / 2 * k21, x2 + dt / 2 * k22, x3 + dt / 2 * k23);
        float k33 = function3(x1 + dt / 2 * k21, x2 + dt / 2 * k22, x3 + dt / 2 * k23);

        float k41 = function1(x1 + dt * k31, x2 + dt * k32, x3 + dt * k33);
        float k42 = function2(x1 + dt * k31, x2 + dt * k32, x3 + dt * k33);
        float k43 = function3(x1 + dt * k31, x2 + dt * k32, x3 + dt * k33);

        x1 = x1 + dt / 6.0 * (k11 + 2 * k21 + 2 * k31 + k41);
        x2 = x2 + dt / 6.0 * (k12 + 2 * k22 + 2 * k32 + k42);
        x3 = x3 + dt / 6.0 * (k13 + 2 * k23 + 2 * k33 + k43);

        diffEq1.push_back(x1);
        diffEq2.push_back(x2);
        diffEq3.push_back(x3);
        time.push_back(t);
    }

    return std::make_tuple(diffEq1, diffEq2, diffEq3, time);
}

//------------------------------------------------------------------------------------

void plot(std::vector<float> t1, std::vector<float> t2, std::vector<float> t3, std::vector<float> time )
{


    std::vector<float> t1mod;
    std::vector<float> t2mod;
    std::vector<float> t3mod;

    for (int ii = 0; ii < t1.size(); ii++)
    {
        t1mod.push_back(fmod(t1[ii], 2 * M_PI));
        t2mod.push_back(fmod(t2[ii], 2 * M_PI));
        t3mod.push_back(fmod(t3[ii], 2 * M_PI));
    }

    plt::title("Three Periodic Signals from Kuramoto Model - First-order Differential Equations");
    plt::named_plot("neuron 1", time, t1mod);
    plt::named_plot("neuron 2", time, t2mod);
    plt::named_plot("neuron 3", time, t3mod);
    plt::xlabel("time");
    plt::ylabel("Y");
    plt::legend();

    plt::show();
}

//------------------------------------------------------------------------------------

int main()
{

    auto cpg = methodRungeKutta1Diff();

    plot(std::get<0>(cpg), std::get<1>(cpg), std::get<2>(cpg), std::get<0>(cpg));
}