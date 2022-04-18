#include"TFIM1d.h"

#include<algorithm>
#include<complex>
#include<exception>
#include<iostream>
#include<stack>
#include<string>
#include<vector>

#include<arpack++/arcomp.h>
#include<arpack++/arlnsmat.h>
#include<arpack++/arlscomp.h>
#include<Eigen/Sparse>
#include<Eigen/Dense>
#include<mpi.h>

void TFIM1d::solve(int Nsites, double J, double g, int nev){
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    if(Nsites >= 50){
        if(world_rank == 0)
            std::cerr << "Nsites is too large (should be less than 64)!" << std::endl;
        throw(std::runtime_error{"Nsites is too large (should be less than 64)!"});
    }
    Nsites_ = Nsites; J_ = J; g_ = g;
    nev_ = nev;

    list_results.clear();
    // loweset_energies.clear();

    generateBasis();
    if(world_rank == 0)
        std::cout << "Number of principle basis is " << list_principle_basis.size() << std::endl;

    //assign works to threads
    int numk_each_process = Nsites / world_size;
    int mark = Nsites % world_size;

    int fromk, endk;
    if(world_rank < mark){
        fromk = (numk_each_process + 1) * world_rank;
        endk = fromk + numk_each_process + 1;
    }    
    else{
        fromk = world_rank * numk_each_process + mark;
        endk = fromk + numk_each_process;
    }

    for(int k = fromk; k < endk; ++k){
        solveAtK(k);
    }

    if(endk > fromk)
        std::cout << "processor " << world_rank << " completed (k=" << fromk << ".." << endk-1 << ")!" << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
}

void TFIM1d::generateBasis(){
    list_basis.clear();
    list_principle_basis.clear();
    uint64_t sz = 1;
    sz <<= Nsites_;
    list_basis.resize(sz);
    Basis temp_basis{0, 0, 0};
    std::fill(list_basis.begin(), list_basis.end(), temp_basis);

    bit_at_left_most = sz >> 1;
    std::stack<uint64_t> sta;
    for(uint64_t i = 0; i < sz; ++i){
        auto &x = list_basis[i];
        if(!x.done){
            list_principle_basis.push_back(i);
            record_principle_basis_index_in_principle_list[i] = list_principle_basis.size()-1;
            x.done = true;
            uint64_t y = i;
            for(int r = 1; r <= Nsites_; ++r){
                y = inverse_translation(y);
                auto &y_basis = list_basis[y]; 
                y_basis.done = true;
                y_basis.rstep_to_principle = r;
                y_basis.principle_basis_index = i;
                if(y == i) break;
            }
        }
    }
}

void TFIM1d::solveAtK(int k){
    ResultsInKSubspace cur_k_space;

    int id = 0;
    cur_k_space.k = k;
    for(auto& x: list_principle_basis){
        if(k * list_basis[x].rstep_to_principle % Nsites_)
            continue;
        cur_k_space.list_principle_basis_index.emplace_back(x);
        cur_k_space.record_index_in_list[x] = id++;
    }

    std::vector<Eigen::Triplet<arcomplex<double>>> list_triplets;
    for(auto& r_basis: cur_k_space.list_principle_basis_index){
        double coeff; uint64_t l_basis;
        std::complex<double> value;
        for(int i = 0; i < Nsites_; ++i){
            l_basis = Operator_J_szsz(i, r_basis, coeff);
            SetMatrixElements(list_triplets, l_basis, r_basis,
                              coeff, cur_k_space);
            l_basis = Operator_g_sx(i, r_basis, coeff);
            SetMatrixElements(list_triplets, l_basis, r_basis,
                              coeff, cur_k_space);
        }
    }

    SolveEigProblem(list_triplets, k, cur_k_space);

    list_results.emplace_back(std::move(cur_k_space));
}

void TFIM1d::SolveEigProblem(std::vector<Eigen::Triplet<arcomplex<double>>>& list_triplets,
                             int k, ResultsInKSubspace& result){
    int matsize = result.list_principle_basis_index.size();
    Eigen::SparseMatrix<arcomplex<double>> spMat(matsize, matsize);
    spMat.setFromTriplets(list_triplets.begin(), list_triplets.end());
    if(matsize <= 100){
        //Dense matrix
        Eigen::MatrixXcd dense_mat;
        dense_mat = spMat.toDense();
        Eigen::SelfAdjointEigenSolver<decltype(dense_mat)> eig_solver(dense_mat);
        result.eigenvalues = eig_solver.eigenvalues();
        result.eigenvectors = eig_solver.eigenvectors();
    }
    else{
        //Sparse matrix
        ARluNonSymMatrix<arcomplex<double>, double> ar_mat(matsize,
                                                            spMat.nonZeros(),
                                                           spMat.valuePtr(),
                                                           spMat.innerIndexPtr(),
                                                           spMat.outerIndexPtr());

        ARluCompStdEig<double> ar_solver(nev_, ar_mat, (char *)("SR"));
        arcomplex<double> *eigvalues = new arcomplex<double>[nev_];
        arcomplex<double> *eigvectors = new arcomplex<double>[matsize * nev_];
        ar_solver.EigenValVectors(eigvectors, eigvalues);

        result.eigenvalues = Eigen::Map<Eigen::VectorXcd>(eigvalues, nev_).real();
        result.eigenvectors = Eigen::Map<Eigen::MatrixXcd>(eigvectors, matsize, nev_);

        delete[] eigvalues;
        delete[] eigvectors;
    }  
}

void TFIM1d::printsz_core(std::ostream& out){
    std::vector<std::vector<int>> list_sz_principle_basis(list_principle_basis.size());
    for(int i = 0; i < list_principle_basis.size(); ++i){
        list_sz_principle_basis[i].resize(Nsites_);
        for(int j = 0; j < Nsites_; ++j){
            int r = list_basis[list_principle_basis[i]].rstep_to_principle;
            int val = 0;
            for(int t = 0; t < r; ++t){
                val += ((list_principle_basis[i] >> j--) & 1)?1:-1;
                j = (j + Nsites_) % Nsites_;
            }
            list_sz_principle_basis[i][j] = val;
        }
    }
    for(auto& cur: list_results){
        out << "k = " << cur.k << "\n";
        int matsize = cur.list_principle_basis_index.size();
        for(int Eindex = 0; Eindex < cur.eigenvalues.rows(); ++Eindex){
            out << "E = " << cur.eigenvalues(Eindex);
            out << "\nsite\tdensite\n";
            for(int j = 0; j < Nsites_; ++j){
                double density = 0;
                for(int phi = 0; phi < matsize; ++phi){
                    density += std::norm(cur.eigenvectors(phi, Eindex))
                               * list_sz_principle_basis[record_principle_basis_index_in_principle_list[\
                                 cur.list_principle_basis_index[phi]]][j]
                               / list_basis[cur.list_principle_basis_index[phi]].rstep_to_principle;
                }
                out << j << " " << density << "\n";
            }
        }
    }
}

void TFIM1d::MPI_PrintToFile(const std::string& filename, void(*func)(TFIM1d *obj, const std::string& filename)){
    MPI_Barrier(MPI_COMM_WORLD);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    bool outputflag;
    if(world_rank == 0){
        std::ofstream fout{filename};
        fout.close();
        func(this, filename);
        outputflag = true;
    }
    else{
        MPI_Recv(&outputflag, 1, MPI_CXX_BOOL, world_rank-1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        func(this, filename);
    }

    if(world_rank < world_size - 1)
        MPI_Send(&outputflag, 1, MPI_CXX_BOOL, world_rank+1, 1, MPI_COMM_WORLD);

}