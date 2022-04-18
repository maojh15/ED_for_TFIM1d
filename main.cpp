#include"TFIM1d.h"

#include<iostream>
#include<fstream>

#include<myTools/ReadPyConfigFile.h>
#include<mpi.h>
using namespace std;

int main(int argc, char** argv){

    MPI_Init(nullptr, nullptr);

    int Nsite;
    double J, g;
    int nev;
    bool output_wavefunction = false;
    bool output_sigma_z = false;
    
    ReadPyConfigFile readPy{"config"};
    bool read_flag = true;
    read_flag &= READPY_VARIABLE(readPy, Nsite);
    read_flag &= READPY_VARIABLE(readPy, J);
    read_flag &= READPY_VARIABLE(readPy, g);
    read_flag &= READPY_VARIABLE(readPy, nev);
    // read_flag &= READPY_VARIABLE(readPy, num_of_threads);
    read_flag &= READPY_VARIABLE_OPTIONAL(readPy, output_wavefunction);
    read_flag &= READPY_VARIABLE_OPTIONAL(readPy, output_sigma_z);
    if(!read_flag) {MPI_Finalize(); return -1;}

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if(world_rank == 0)
        cout << "Number of processors: " << world_size << std::endl;

    TFIM1d solver;
    if(world_rank == 0)
        cout << "computing ... " << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    
    solver.solve(Nsite, J, g, nev);

    if(world_rank == 0)
        cout << "writing energy to file ... " << std::endl;
    solver.PrintEnergySpectrum("fout_energy.txt");

    if(output_wavefunction){
        if(world_rank == 0)
            cout << "writing wavefunction to file ... " << std::endl;
        solver.PrintWavefunction("fout_wavefunction.txt");
    }
    else{
        if(world_rank == 0){
            std::ofstream{"fout_wavefunction.txt"};
        }
    }

    if(output_sigma_z){
        if(world_rank == 0)
            cout << "writing sigma^z to file ..." << std::endl;
        solver.PrintSigma_z("fout_sigma_z.txt");
    }
    else{
        if(world_rank == 0){
            std::ofstream{"fout_sigma_z.txt"};
        }
    }

    MPI_Finalize();
    return 0;
}