#include <rvv/rvv.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <concepts>

bool test_true(bool x){
    if(!x){
        std::cout << "Test failed!" << std::endl;
    }
    return x;
}

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

    bool success = std::equal(data.begin(), data.end(), x_data.begin());

    return test_true(success);
}

template <typename T>
bool test(){
    bool success = true;
    using namespace rvv::experimental;
    const int simd_size = simd<T>::size();

    std::vector<bool> _random_mask(simd_size);
    std::vector<bool> _random_mask2(simd_size);

    for (int i = 0; i < simd_size; i++){
        _random_mask[i] = std::rand() % 2 == 0;
        _random_mask2[i] = std::rand() % 2 == 0;
    }

    const std::vector<bool> random_mask = _random_mask;
    const std::vector<bool> random_mask2 = _random_mask2;

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
            x.set(i, random_mask[i]);
        }
        std::cout << "x (set): " << x << std::endl;
        for (int i = 0; i < simd_size; i++){
            success &= test_true(x[i] == random_mask[i]);
            success &= test_true(x.get(i) == random_mask[i]);
        }
    }
    {
        // operators
        simd_mask<T> x1, x2;
        for (size_t i = 0; i < simd_size; i++)
        {
            x1.set(i, random_mask[i]);
            x2.set(i, random_mask2[i]);
        }
        std::cout << "x1: " << x1 << std::endl;
        std::cout << "x2: " << x2 << std::endl;

        simd_mask<T> x_not = !x1;
        std::cout << "!x1: " << x_not << std::endl;
        for (size_t i = 0; i < simd_size; i++)
            success &= test_true(x_not[i] == !random_mask[i]);
        

        simd_mask<T> x_and = x1 && x2;
        std::cout << "x1 && x2: " << x_and << std::endl;
        for (size_t i = 0; i < simd_size; i++)
            success &= test_true(x_and[i] == (random_mask[i] && random_mask2[i]));


        simd_mask<T> x_or = x1 || x2;
        std::cout << "x1 || x2: " << x_or << std::endl;
        for (size_t i = 0; i < simd_size; i++)
            success &= test_true(x_or[i] == (random_mask[i] || random_mask2[i]));

        simd_mask<T> x_xor = x1 ^ x2;
        std::cout << "x1 ^ x2: " << x_xor << std::endl; 
        for (size_t i = 0; i < simd_size; i++)
            success &= test_true(x_xor[i] == (random_mask[i] ^ random_mask2[i]));
  
    }

    // choose / mask_assign
    {
        simd_mask<T> x1(false);
        simd<T> v1, v2;
        for (size_t i = 0; i < simd_size; i++)
        {
            x1.set(i, random_mask[i]);
            v1.set(i, T(std::rand() % 100));
            v2.set(i, T(std::rand() % 100));
        }

        std::cout << "choose: " << std::endl;
        std::cout << "x1: " << x1 << std::endl;
        std::cout << "v1: " << v1 << std::endl;
        std::cout << "v2: " << v2 << std::endl;
        simd<T> v3 = choose(x1, v1, v2);
        std::cout << "v3: " << v3 << std::endl;
        
        for (size_t i = 0; i < simd_size; i++){
            if (x1[i]){
                success &= test_true(v3[i] == v1[i]);
            } else {
                success &= test_true(v3[i] == v2[i]);
            }
        }

        simd<T> v1_old = v1;
        std::cout << "mask_assign: " << std::endl;
        std::cout << "x1: " << x1 << std::endl;
        std::cout << "v1_old: " << v1_old << std::endl;
        std::cout << "v2: " << v2 << std::endl;
        mask_assign(x1, v1, v2);
        std::cout << "v1: " << v1 << std::endl;

        for (size_t i = 0; i < simd_size; i++){
            if (x1[i]){
                success &= test_true(v1[i] == v2[i]);
            } else {
                success &= test_true(v1[i] == v1_old[i]);
            }
        }
        
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