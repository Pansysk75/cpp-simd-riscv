#include <rvv/rvv.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

template <typename T>
bool test_equal(rvv::experimental::simd<T> x, const std::vector<T>& data){

    std::cout << "x:    " << x << std::endl;
    std::cout << "data: ( ";
    for(auto el : data){
        std::cout << int(el) << " ";
    }
    std::cout << ")\n" << std::endl;
    std::vector<T> x_data(x.size());
    x.copy_to(x_data.data(), rvv::experimental::vector_aligned);

    return std::equal(data.begin(), data.end(), x_data.begin());
}

template <typename T>
bool test(){
    bool success = true;
    using namespace rvv::experimental;
    const int simd_size = simd<T>::size();

    {
        // Default constructor (all 0)
        simd_mask<T> x1;
        std::cout << "x1 (default -> false): " << x1 << std::endl;
        for (int i = 0; i < simd_size; i++){
            success &= !x1[i];
        }

        // Constructor from bool
        simd_mask<T> x2(true);
        std::cout << "x2 (true): " << x2 << std::endl;
        simd_mask<T> x3(false);
        std::cout << "x3 (false): " << x3 << std::endl;
        for (int i = 0; i < simd_size; i++){
            success &= x2[i];
            success &= !x3[i];
        }
    }

    {
        // set-get
        simd_mask<T> x;
        for (int i = 0; i < simd_size; i++){
            x.set(i, i % 2 == 0);
        }
        std::cout << "x (set): " << x << std::endl;
        for (int i = 0; i < simd_size; i++){
            success &= x[i] == (i % 2 == 0);
        }
    }
    {
        // operators
        simd_mask<T> x1, x2;
        for (size_t i = 0; i < simd_size; i++)
        {
            x1.set(i, i % 2 == 0);
            x2.set(i, i % 3 == 0);
        }
        std::cout << "x1: " << x1 << std::endl;
        std::cout << "x2: " << x2 << std::endl;

        simd_mask<T> x_not = !x1;
        std::cout << "!x1: " << x_not << std::endl;
        simd_mask<T> x_and = x1 && x2;
        std::cout << "x1 && x2: " << x_and << std::endl;
        simd_mask<T> x_or = x1 || x2;
        std::cout << "x1 || x2: " << x_or << std::endl;
        simd_mask<T> x_xor = x1 ^ x2;
        std::cout << "x1 ^ x2: " << x_xor << std::endl; 

        for (size_t i = 0; i < simd_size; i++)
        {
            success &= x_not[i] == !(i % 2 == 0);
            success &= x_and[i] == (i % 2 == 0 && i % 3 == 0);
            success &= x_or[i] == (i % 2 == 0 || i % 3 == 0);
            success &= x_xor[i] == (i % 2 == 0) ^ (i % 3 == 0);
        }
    }

    // choose
    {
        simd_mask<T> x1(false);
        simd<T> v1, v2;
        for (size_t i = 0; i < simd_size; i++)
        {
            x1.set(i, i % 2 == 0);
            v1.set(i, i);
            v2.set(i, simd_size-i);
        }

        simd<T> v3 = choose(x1, v1, v2);
        
        for (size_t i = 0; i < simd_size; i++){
            if(i%2){
                success &= (v3[i] == i);
            }else{
                success &= (v3[i] == simd_size-i);
            }
        }

        mask_assign(x1, v1, v2);

        success &= (v1 == v3).all_of();
    }
    
    return success;
}

int main(){
    bool success = true;

    std::cout << "\nTesting type: " << "int8_t" << std::endl;
    success &= test<int8_t>();
    std::cout << "\nTesting type: " << "int16_t" << std::endl;
    success &= test<int16_t>();
    std::cout << "\nTesting type: " << "int32_t" << std::endl;
    success &= test<int32_t>();
    std::cout << "\nTesting type: " << "int64_t" << std::endl;
    success &= test<int64_t>();

    std::cout << "\nTesting type: " << "uint8_t" << std::endl;
    success &= test<uint8_t>();
    std::cout << "\nTesting type: " << "uint16_t" << std::endl;
    success &= test<uint16_t>();
    std::cout << "\nTesting type: " << "uint32_t" << std::endl;
    success &= test<uint32_t>();
    std::cout << "\nTesting type: " << "uint64_t" << std::endl;
    success &= test<uint64_t>();

    std::cout << "\nTesting type: " << "float" << std::endl;
    success &= test<float>();
    std::cout << "\nTesting type: " << "double" << std::endl;
    success &= test<double>();

    return success ? 0 : -1;
}