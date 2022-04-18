#ifndef _TFIM1d_H_
#define _TFIM1d_H_

#include<complex>
#include<fstream>
#include<set>
#include<string>
#include<unordered_map>
#include<vector>

#include<arpack++/arcomp.h>
#include<Eigen/Dense>
#include<Eigen/Sparse>


class TFIM1d{
public:
    /**
     * @brief Exact diagonalization of 1d transverse field Ising model
     * 
     * @param Nsites number of lattice sites
     * @param J exchange coefficient
     * @param g external field along x-direction
     * @param numOfThreads number of threads, default to be 1.
     * @param nev number of eigenvalues to be solved.
     * @return true 
     * @return gap 
     */
    void solve(int Nsites, double J, double g, int nev = 10);

    struct ResultsInKSubspace{
        int k;
        std::vector<uint64_t> list_principle_basis_index;
        std::unordered_map<uint64_t, std::size_t> record_index_in_list;
        Eigen::VectorXd eigenvalues;
        Eigen::MatrixXcd eigenvectors;
    };

    std::vector<ResultsInKSubspace> list_results;
    // double gap, ground_state_energy;
    // std::set<double> loweset_energies;

    void PrintEnergySpectrum(const std::string& filename){
        this->MPI_PrintToFile(filename, printeigvals);
    }

    void PrintWavefunction(const std::string& filename){
        this->MPI_PrintToFile(filename, printeigvecs);
    }
    
    void PrintSigma_z(const std::string& filename){
        this->MPI_PrintToFile(filename, printsz);
    }

    double epsilon_generacy = 0.01;

private:
    int Nsites_;
    double J_, g_;
    int nev_;

    void solveAtK(int k); //k = 0,1,2,...,Nsite-1

    void generateBasis();

    struct Basis{
        int rstep_to_principle = 0;
        uint64_t principle_basis_index = 0;
        bool done = false;
    };
    std::vector<Basis> list_basis;
    std::vector<uint64_t> list_principle_basis;
    std::unordered_map<uint64_t, std::size_t> record_principle_basis_index_in_principle_list;
    
    //translate to right
    uint64_t translation(uint64_t basis){
        bool lastBit = basis & 1;
        basis >>= 1;
        if(lastBit){
            basis &= bit_at_left_most;
        }
        return basis;
    }

    //translate to left
    uint64_t inverse_translation(uint64_t basis){
        uint64_t left_most_bit = basis & bit_at_left_most;
        if(left_most_bit) basis -= bit_at_left_most;
        basis <<= 1;
        if(left_most_bit) basis += 1;
        return basis;
    }

    uint64_t bit_at_left_most;

    /**
     * @brief Action of operator \f$-J_\sigma^z_i\sigma^z_{i+1}\f$
     * 
     * @param site_index the index \f$i\f$.
     * @param basis 
     * @return uint64_t 
     */
    uint64_t Operator_J_szsz(int site_index, uint64_t basis, double& coeff){
        uint64_t l_basis = basis;
        if(site_index == Nsites_ - 1){
            coeff = (basis & 1)? 1: -1;
            coeff *= ((basis >> site_index)&1) ? 1: -1;
        }
        else{
            basis >>= site_index;
            coeff = (basis & 1)? 1: -1;
            coeff *= ((basis >> 1) & 1)? 1: -1;
        }
        coeff *= -J_;
        return l_basis;
    }

    uint64_t Operator_g_sx(int site_index, uint64_t basis, double& coeff){
        uint64_t mask = 1;
        mask <<= site_index;
        basis ^= mask;
        coeff = -g_;
        return basis;     
    }

    void SetMatrixElements(std::vector<Eigen::Triplet<arcomplex<double>>>& list_triplets,
                           uint64_t l_basis, uint64_t r_basis, double coeff, ResultsInKSubspace& cur_k_space){
        int k = cur_k_space.k;
        auto& record = cur_k_space.record_index_in_list;
        if(record.find(l_basis) == record.end()) return;

        using namespace std::complex_literals;
        std::complex<double> value = coeff;
        int lj = list_basis[l_basis].rstep_to_principle;
        l_basis = list_basis[l_basis].principle_basis_index;
        double Al = list_basis[l_basis].rstep_to_principle;
        double Ar = list_basis[r_basis].rstep_to_principle;
        value *= std::sqrt(Ar / Al) * std::exp(-1.0i * PI2 * (static_cast<double>(k * lj) / Nsites_));
        list_triplets.emplace_back(record[l_basis], record[r_basis], value);
    }

    void SolveEigProblem(std::vector<Eigen::Triplet<arcomplex<double>>>& list_triplets,
                         int k, ResultsInKSubspace& result);

    const double PI2 = 6.2831853071795864769;

    void MPI_PrintToFile(const std::string& filename, 
                         void(*func)(TFIM1d *obj, const std::string& filename));

    static void printeigvals(TFIM1d* obj, const std::string& filename){
        std::ofstream fout{filename, std::ios::app};
        fout.precision(15);
        for(auto& x: obj->list_results){
            fout << "k = " << x.k << " dim = " << x.list_principle_basis_index.size() <<  "\n";
            fout << x.eigenvalues << "\n";
        }
        fout.close();
    }

    static void printeigvecs(TFIM1d* obj, const std::string& filename){
        std::ofstream fout_wavefunction(filename, std::ios::app);
        fout_wavefunction.precision(15);
        for(auto& list: obj->list_results){
            fout_wavefunction << "k = " << list.k << "\n";
            fout_wavefunction << "principle symmetric basis:\n";
            for(auto& x: list.list_principle_basis_index){
                fout_wavefunction << x << " ";
            }
            fout_wavefunction << "\n";
            for(int i = 0; i < list.eigenvalues.rows(); ++i){
                fout_wavefunction << "E = " << list.eigenvalues(i)
                                << "\n" << list.eigenvectors.col(i) << "\n";            
            } 
        }
        fout_wavefunction.close();
    }

    void printsz_core(std::ostream& out);

    static void printsz(TFIM1d* obj, const std::string& filename){
        std::ofstream fout{filename, std::ios::app};
        fout.precision(15);
        obj->printsz_core(fout);
        fout.close();
    }
};

#endif