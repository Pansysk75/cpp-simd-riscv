#include <sve/sve.hpp>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

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
    simd<T> x;
    std::vector<T> data(x.size());
    std::iota(data.begin(), data.end(), 5);
    
    std::cout << "Copy from simd" << std::endl;
    x.copy_from(data.data(), vector_aligned);
    success &= test_equal(x, data);

    std::cout << "Copy to simd" << std::endl;
    std::fill(data.begin(), data.end(), 0);
    x.copy_to(data.data(), vector_aligned);
    success &= test_equal(x, data);

    std::cout << "Initialization from simd" << std::endl;
    simd<T> y(x);
    success &= test_equal(y, data);

    std::cout << "Initialization from scalar" << std::endl;    
    std::fill(data.begin(), data.end(), 42);
    x = simd<T>(42);
    success &= test_equal(x, data);

    }

    {
    std::vector<T> data(simd_size);
    std::iota(data.begin(), data.end(), 5);
    simd<T> x(data.data(), vector_aligned);

    std::cout << "Addition" << std::endl;
    //
    std::for_each(data.begin(), data.end(), [](auto& elem){ elem += 5;});
    x += 5;
    success &= test_equal(x, data);
    //
    std::for_each(data.begin(), data.end(), [](auto& elem){ elem += 3;});
    x = x + 3;
    success &= test_equal(x, data);
    //
    std::for_each(data.begin(), data.end(), [](auto& elem){ elem += 1;});
    success &= test_equal(++x, data);
    success &= test_equal(x++, data);
    }

    {
    std::vector<T> data(simd_size);
    std::iota(data.begin(), data.end(), 5);
    simd<T> x(data.data(), vector_aligned);
    std::cout << "Subtraction" << std::endl;
    //
    std::for_each(data.begin(), data.end(), [](auto& elem){ elem -= 5;});
    x -= 5;
    success &= test_equal(x, data);
    //
    std::for_each(data.begin(), data.end(), [](auto& elem){ elem -= 3;});
    x = x - 3;
    success &= test_equal(x, data);
    //
    std::for_each(data.begin(), data.end(), [](auto& elem){ elem -= 1;});
    success &= test_equal(--x, data);
    success &= test_equal(x--, data);
    }


    
    return success;
}

int main(){
    bool success = true;

    success &= test<int8_t>();
    success &= test<int16_t>();

    success &= test<uint8_t>();
    success &= test<uint16_t>();

    return success ? 0 : -1;
}