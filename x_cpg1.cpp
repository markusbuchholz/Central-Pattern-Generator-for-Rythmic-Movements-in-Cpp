// Markus Buchholz, 2023
// g++ x_cpg1.cpp -o t -I/usr/include/python3.8 -lpython3.8

#include <iostream>
#include <vector>
#include <random>
#include <math.h>
#include <cmath>

#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

int iterations = 1000;
float K = 1.0f;
float dt = 0.01f;
float omega = 2.0f * M_PI;

//------------------------------------------------------------------------------------
float generateRandom()
{

    std::random_device engine;
    std::mt19937 gen(engine());
    std::uniform_real_distribution<float> distribution(0, 2 * M_PI);

    return distribution(gen);
}

//------------------------------------------------------------------------------------

struct Neuron
{

    std::vector<float> phases;
    float init;
};

//------------------------------------------------------------------------------------

class CPG
{

public:
    int nr_neurones;
    std::vector<Neuron> gen;

public:
    CPG(int n) : nr_neurones(n)
    {

        std::cout << "init neurones "
                  << "\n";

        for (int ii = 0; ii < nr_neurones; ii++)
        {

            Neuron neuron;
            neuron.init = generateRandom();
            neuron.phases.push_back(neuron.init);
            gen.push_back(neuron);
        }
    }

    // Kuramoto model
    void computeModel()
    {

        for (int t = 1; t < iterations; t++)
        {
            float delta_phase = 0.0f;

            for (int ii = 0; ii < nr_neurones; ii++)
            {
                for (int jj = 0; jj < nr_neurones; jj++)
                {

                    delta_phase += (K / nr_neurones) * std::sin(gen[jj].phases[t - 1] - gen[ii].phases[t - 1]);
                }

                gen[ii].phases.push_back(gen[ii].phases[t - 1] + omega * dt + delta_phase * dt);
            }
        }
    }

    void plot()
    {

        std::vector<int> time;
        for (int ii = 0; ii < iterations; ii++)
        {
            time.push_back(ii);
        }

        std::vector<float> t1 = gen[0].phases;
        std::vector<float> t2 = gen[1].phases;
        std::vector<float> t3 = gen[2].phases;
        std::vector<float> t1mod;
        std::vector<float> t2mod;
        std::vector<float> t3mod;

        for (int ii = 0; ii < t1.size(); ii++)
        {
            t1mod.push_back(fmod(t1[ii], 2 * M_PI));
            t2mod.push_back(fmod(t2[ii], 2 * M_PI));
            t3mod.push_back(fmod(t3[ii], 2 * M_PI));
        }

        plt::title("CPG. Kuramoto model ");
        plt::named_plot("neuron 1", time, t1mod);
        plt::named_plot("neuron 2", time, t2mod);
        plt::named_plot("neuron 3", time, t3mod);
        plt::xlabel("time");
        plt::ylabel("Y");
        plt::legend();

        plt::show();
    }
};

//------------------------------------------------------------------------------------

int main()
{

    CPG cpg(3);

    cpg.computeModel();
    cpg.plot();
}