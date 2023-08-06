// Markus Buchholz, 2023
// g++ x_cpg2.cpp -o t -I/usr/include/python3.8 -lpython3.8

#include <iostream>
#include <vector>
#include <random>
#include <math.h>
#include <cmath>
#include <algorithm>

#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

// Hodgkin-Huxley model parameters
float C_m = 1.0f;     // Membrane capacitance (in µF/cm^2)
float g_Na = 120.0f;  // Sodium conductance (in mS/cm^2)
float g_K = 36.0f;    // Potassium conductance (in mS/cm^2)
float g_L = 0.3f;     // Leak conductance (in mS/cm^2)
float E_Na = 50.0f;   // Sodium reversal potential (in mV)
float E_K = -77.0f;   // Potassium reversal potential (in mV)
float E_L = -54.387f; // Leak reversal potential (in mV)
#include <cmath>
//------------------------------------------------------------------------------------
float generateRandom(float a, float b)
{

    std::random_device engine;
    std::mt19937 gen(engine());
    std::uniform_real_distribution<float> distribution(a, b);

    return distribution(gen);
}
//------------------------------------------------------------------------------------
// Helper function to compute gating variables
float alpha_m(float V)
{

    return 0.1 * (V + 40.0f) / (1.0f - std::exp(-(V + 40.0f) / 10.0f));
}

float beta_m(float V)
{

    return 4.0f * std::exp(-(V + 65.0f) / 18.0f);
}

float alpha_h(float V)
{

    return 0.07f * std::exp(-(V + 65.0f) / 20.0f);
}

float beta_h(float V)
{

    return 1.0f / (1.0f + std::exp(-(V + 35.0f) / 10.0f));
}

float alpha_n(float V)
{

    return 0.01f * (V + 55.0f) / (1.0f - std::exp(-(V + 55.0f) / 10.0f));
}

float beta_n(float V)
{

    return 0.125f * std::exp(-(V + 65.0f) / 80.0f);
}

// Generate random coupling strengths between neurons
std::vector<float> gen_kx()
{

    std::vector<float> kx = []()
    {
        std::vector<float> k_i(4); // Initialize with 5 elements, all set to 0.0

        std::generate(k_i.begin(), k_i.end(), []()
                      { return generateRandom(0.1f, 1.5f); });

        return k_i;
    }();
    return kx;
}

//------------------------------------------------------------------------------------

// Hodgkin-Huxley model simulation
std::tuple<float, float, float, float> hodgkin_huxley_model(float V, float m, float h, float n, float I_ext)
{

    float V_m = V;

    float dV_m = (I_ext - g_Na * m * m * m * h * (V_m - E_Na) - g_K * n * n * n * n * (V_m - E_K) - g_L * (V_m - E_L)) / C_m;
    float dm = alpha_m(V_m) * (1.0 - m) - beta_m(V_m) * m;
    float dh = alpha_h(V_m) * (1.0 - h) - beta_h(V_m) * h;
    float dn = alpha_n(V_m) * (1.0 - n) - beta_n(V_m) * n;

    return std::make_tuple(dV_m, dm, dh, dn);
}

//------------------------------------------------------------------------------------

// Parameters for the Kuramoto model
int num_neurons = 4;
int num_steps = 1000;
float dt = 0.01;
float noise_intensity = 0.2f; // Intensity of noise

// Initialize the vector before main using global variable and a lambda function
std::vector<float> omega = []()
{
    std::vector<float> omega_i(num_neurons); // Initialize with 5 elements, all set to 0.0

    std::generate(omega_i.begin(), omega_i.end(), []()
                  { return generateRandom(1.0f, 3.0f); });

    return omega_i;
}();

// Generate random coupling strengths between neurons

std::vector<std::vector<float>> K_matrix = {gen_kx(), gen_kx(), gen_kx(), gen_kx()};

std::vector<std::vector<float>> K_diag = []()
{
    int kk = 0;
    for (int ii = 0; ii < num_neurons; ii++)
    {
        for (int jj = 0; jj < K_matrix[0].size(); jj++)
        {
            K_matrix[ii][kk] = 0.0f;
        }
        kk++;
    }
    return K_matrix;
}();

//------------------------------------------------------------------------------------

std::tuple<std::vector<float>, std::vector<std::vector<float>>> computeCFG()
{

    std::vector<float> time;

    // Initial membrane potentials and gating variables for each neuron
    std::vector<float> initial_V = []()
    {
        std::vector<float> v_i(num_neurons); // Initialize with 5 elements, all set to 0.0

        std::generate(v_i.begin(), v_i.end(), []()
                      { return generateRandom(-70.0f, -50.0f); });

        return v_i;
    }();

    std::vector<float> initial_m = initial_V;

    std::for_each(initial_m.begin(), initial_m.end(), [](float &value)
                  { value = alpha_m(value) / (alpha_m(value) + beta_m(value)); });

    std::vector<float> initial_h = initial_V;

    std::for_each(initial_h.begin(), initial_h.end(), [](float &value)
                  { value = alpha_h(value) / (alpha_h(value) + beta_h(value)); });

    std::vector<float> initial_n = initial_V;

    std::for_each(initial_n.begin(), initial_n.end(), [](float &value)
                  { value = alpha_n(value) / (alpha_n(value) + beta_n(value)); });

    std::vector<float> zeros(num_steps, 0.0f);
    std::vector<std::vector<float>> V(num_neurons, zeros);
    std::vector<std::vector<float>> m(num_neurons, zeros);
    std::vector<std::vector<float>> h(num_neurons, zeros);
    std::vector<std::vector<float>> n(num_neurons, zeros);
    std::vector<std::vector<float>> phases(num_neurons, zeros);

    // Initialize neuron states with initial values
    V[0] = initial_V;
    m[0] = initial_m;
    h[0] = initial_h;
    n[0] = initial_n;

    // Kuramoto model simulation with Hodgkin-Huxley neurons and noise

    time.push_back(0);
    for (int t = 1; t < num_steps; t++)
    {

        std::vector<float> delta_phases(num_neurons, 0.0f);
        std::vector<float> I_synaptic(num_neurons, 0.0f);

        for (int ii = 0; ii < num_neurons; ii++)
        {
            for (int jj = 0; jj < num_neurons; jj++)
            {
                I_synaptic[ii] += K_diag[ii][jj] * std::sin(phases[jj][t - 1] - phases[ii][t - 1]);
            }
        }

        for (auto &ii : I_synaptic){
            std::cout << ii << " ,";
        }

        std::cout << "\n";

        for (int ii = 0; ii < num_neurons; ii++)
        {

            auto model = hodgkin_huxley_model(V[ii][t - 1], m[ii][t - 1], h[ii][t - 1], n[ii][t - 1], I_synaptic[ii] + noise_intensity * generateRandom(0.0f, 1.0f));

            V[ii][t] = V[ii][t - 1] + dt * std::get<0>(model);
            m[ii][t] = m[ii][t - 1] + dt * std::get<1>(model);
            h[ii][t] = h[ii][t - 1] + dt * std::get<2>(model);
            n[ii][t] = n[ii][t - 1] + dt * std::get<3>(model);
        }

        // Compute new phase values based on membrane potentials

        for (int ii = 0; ii < delta_phases.size(); ii++)
        {
            delta_phases[ii] = omega[ii] * dt + (V[ii][t] - E_L) * dt;
        }

        for (int ii = 0; ii < phases.size(); ii++)
        {
            phases[ii][t] = std::fmod(phases[ii][t - 1] + delta_phases[ii], 2 * M_PI);
        }

        time.push_back(t + dt);
    }

    return std::make_tuple(time, phases);
}

//------------------------------------------------------------------------------------

void plot(std::vector<float> time, std::vector<std::vector<float>> sim)
{

    std::vector<float> n1 = sim[0];
    std::vector<float> n2 = sim[1];
    std::vector<float> n3 = sim[2];
    std::vector<float> n4 = sim[3];

    plt::title("Kuramoto Model with Hodgkin-Huxley Neurons and Noise");
    plt::named_plot("neuron 1", time, n1);
    plt::named_plot("neuron 2", time, n2);
    plt::named_plot("neuron 3", time, n3);
    plt::named_plot("neuron 4", time, n4);
    plt::xlabel("time");
    plt::ylabel("Y");
    plt::legend();

    plt::show();
}

//------------------------------------------------------------------------------------
int main()
{
    auto simulation = computeCFG();
    plot(std::get<0>(simulation), std::get<1>(simulation));
}