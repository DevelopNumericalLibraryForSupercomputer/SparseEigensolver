#pragma once
#include <iostream>
#include <map>
//#include <yaml-cpp/yaml.h>
#include "Utility.hpp"
namespace SE{


typedef enum{
    Real,
    RealSym,
    Complex,
    Hermitian,
} MAT_TYPE;

typedef enum{
    Diagonal
} PRECOND_TYPE;

class DecomposeOption{
public:
    DecomposeOption() {};
    //DecomposeOption(std::string filename);
    //void set_option(std::string filename);
    //void print();
    DecomposeMethod algorithm_type = Davidson;
    int max_iterations       = 100;
    double tolerance         = 1E-10;
    MAT_TYPE matrix_type     = RealSym;
    int num_eigenvalues      = 3;
    int eigenvalue_guesses   = 0;
    //bool use_preconditioner    = false;
    PRECOND_TYPE preconditioner = Diagonal;
    double preconditioner_tolerance      = 1E-10;
    double preconditioner_max_iterations = 30;

private:
    //void set_option_worker();

    template<typename enum_type> enum_type table_match(std::map<std::string, enum_type> table, std::string str);

    std::map<std::string, DecomposeMethod> const algo_table =
        { {"Direct", DecomposeMethod::Direct}, {"Davidson", DecomposeMethod::Davidson}};//, {"LOBPCG", DecomposeMethod::LOBPCG} };
    std::map<std::string, MAT_TYPE> const mat_table =
        { {"Real", MAT_TYPE::Real}, {"RealSym", MAT_TYPE::RealSym}, {"Complex", MAT_TYPE::Complex},{"Hermitian", MAT_TYPE::Hermitian} };
    std::map<std::string, PRECOND_TYPE> const precond_table =
        { {"Diagonal", PRECOND_TYPE::Diagonal}};
    //YAML::Node config;
};
/*
DecomposeOption::DecomposeOption(){
    config = YAML::LoadFile("Default.yaml");
    set_option_worker();
}

DecomposeOption::DecomposeOption(std::string filename){
    config = YAML::LoadFile("Default.yaml");
    set_option(filename);
}

void DecomposeOption::set_option(std::string filename){

    YAML::Node new_config = YAML::LoadFile(filename);

    for(YAML::const_iterator it1 = new_config.begin() ; it1 != new_config.end() ; ++it1) {
        std::string option_type_string = it1->first.as<std::string>();
        YAML::Node option_type = new_config[option_type_string];

        for(YAML::const_iterator it2 = option_type.begin() ; it2 != option_type.end() ; ++it2) {
            std::string key = it2->first.as<std::string>();      // <- key
            std::string value = it2->second.as<std::string>();         // <- value
            config[option_type_string][key] = value;
        }
    }
    set_option_worker();
}

void DecomposeOption::print(){
    YAML::Emitter emit;
    emit  << config;
    std::cout << "Decomposition options : " << std::endl;
    std::cout << emit.c_str() << std::endl;
}

void DecomposeOption::set_option_worker(){
    this->algorithm_type = table_match<DecomposeMethod>(algo_table, config["solver_options"]["algorithm"].as<std::string>());
    this->max_iterations = config["solver_options"]["max_iterations"].as<int>();
    this->tolerance = config["solver_options"]["tolerance"].as<double>();
    this->matrix_type = table_match<MAT_TYPE>(mat_table, config["matrix_options"]["matrix_type"].as<std::string>());

    this->num_eigenvalues = config["eigenvalue_options"]["num_eigenvalues"].as<int>();
    this->eigenvalue_guesses = config["eigenvalue_options"]["eigenvalue_guesses"].as<int>();

    this->use_preconditioner = config["preconditioner_options"]["use_preconditioner"].as<bool>();
    this->preconditioner_tolerance = config["preconditioner_options"]["preconditioner_tolerance"].as<double>();
    this->preconditioner_max_iterations = config["preconditioner_options"]["preconditioner_max_iterations"].as<double>();
}
*/
template <typename enum_type>
enum_type DecomposeOption::table_match(std::map<std::string, enum_type> table, std::string str){
    if(table.find(str) != table.end()){
        return table.find(str)->second;
    }
    else{
        std::cout << "WRONG OPTION : " << str << std::endl;
        exit(-1);
    }
}

}
/*
int main(){
    TH::DecomposeOption first_option;
    TH::DecomposeOption second_option = TH::DecomposeOption("test.yaml");

    first_option.print();
    std::cout << std::endl;
    second_option.print();
    return 0;
}
*/