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

    std::cout << "simd set - get" << std::endl;
    for(int i = 0; i < simd_size; i++){
        T val = 2*i + 42;
        x.set(i, val);
        success &= (x.get(i) == val);
        success &= (x[i] == val);
        std::cout << "val: " << val << " x.get(i): " << x.get(i) << " x[i]: " << x[i] << std::endl;
    }
    std::cout << std::endl;

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

    
    {
    std::vector<T> data(simd_size);
    std::iota(data.begin(), data.end(), 5);
    simd<T> x(data.data(), vector_aligned);
    std::cout << "Multiplication" << std::endl;
    //
    std::for_each(data.begin(), data.end(), [](auto& elem){ elem *= 5;});
    x *= 5;
    success &= test_equal(x, data);
    //
    std::for_each(data.begin(), data.end(), [](auto& elem){ elem *= -3;});
    x = x * (-3);
    success &= test_equal(x, data);
    //
    std::for_each(data.begin(), data.end(), [](auto& elem){ elem *= -1;});
    success &= test_equal(-x, data);
    }

    {
    std::vector<T> data(simd_size);
    std::iota(data.begin(), data.end(), 120);
    simd<T> x(data.data(), vector_aligned);
    std::cout << "Division" << std::endl;
    //
    std::for_each(data.begin(), data.end(), [](auto& elem){ elem /= 5;});
    x /= 5;
    success &= test_equal(x, data);
    //
    std::for_each(data.begin(), data.end(), [](auto& elem){ elem = elem / T(-3);});
    x = x / (-3);
    success &= test_equal(x, data);
    }

    // Bitwise operations only supported for integral types
    if constexpr(std::is_integral_v<T>)
    {
        std::vector<T> data(simd_size);
        std::iota(data.begin(), data.end(), 5);
        simd<T> x(data.data(), vector_aligned);
        std::cout << "Bitwise operations" << std::endl;
        //
        std::for_each(data.begin(), data.end(), [](auto& elem){ elem &= 0x03;});
        x &= 0x03;
        success &= test_equal(x, data);
        //
        std::for_each(data.begin(), data.end(), [](auto& elem){ elem |= 0xF0;});
        x |= 0xF0;
        success &= test_equal(x, data);
        //
        std::for_each(data.begin(), data.end(), [](auto& elem){ elem ^= 0xFF;});
        x ^= 0xFF;
        success &= test_equal(x, data);
    }

    {
    std::vector<T> data(simd_size);
    std::iota(data.begin(), data.end(), 5);
    simd<T> x(data.data(), vector_aligned);
    std::cout << "Comparison\n" << std::endl;
    //
    simd<T> y(x);
    success &= (x == y).all_of();
    success &= (x != y).none_of();
    success &= (x < y).none_of();
    success &= (x <= y).all_of();
    success &= (x > y).none_of();
    success &= (x >= y).all_of();
    //
    y = simd<T>(data.data(), vector_aligned);
    y += 1;
    success &= (x == y).none_of();
    success &= (x != y).all_of();
    success &= (x < y).all_of();
    success &= (x <= y).all_of();
    success &= (x > y).none_of();
    success &= (x >= y).none_of();
    }

    // Elementwise Binary Algorithms
    {
    std::vector<T> data_x(simd_size), data_y(simd_size), data_res(simd_size);
    
    std::iota(data_x.begin(), data_x.end(), 0);
    std::iota(std::reverse_iterator(data_y.end()), std::reverse_iterator(data_y.begin()), 0);

    simd<T> x(data_x.data(), vector_aligned);
    simd<T> y(data_y.data(), vector_aligned);
    simd<T> res(data_res.data(), vector_aligned);

    std::cout << "Elementwise Binary Algorithms" << std::endl;
    // min
    std::cout << "max: " << std::endl;
    std::transform(data_x.begin(), data_x.end(), data_y.begin(), data_res.begin(), [](auto a, auto b){ return std::min(a, b); });
    success &= test_equal(min(x, y), data_res);
    // max
    std::cout << "max: " << std::endl;
    std::transform(data_x.begin(), data_x.end(), data_y.begin(), data_res.begin(), [](auto a, auto b){ return std::max(a, b); });
    success &= test_equal(max(x, y), data_res);
    }

    // Unary Algorithms
    {
    std::vector<T> data(simd_size);
    std::iota(data.begin(), data.end(), 5);
    simd<T> x(data.data(), vector_aligned);
    std::cout << "Unary Algorithms" << std::endl;
    // abs
    if constexpr(std::is_signed_v<T>){
        std::transform(data.begin(), data.end(), data.begin(), [](const T& a){ return std::abs(a); });
        success &= test_equal(abs(x), data);
    }
    // sqrt
    if constexpr(std::is_floating_point_v<T>)
    {
        std::transform(data.begin(), data.end(), data.begin(), [](auto a){ return std::sqrt(a); });
        success &= test_equal(sqrt(x), data);
    }
    }

    // Reduction algorithms
    {
    std::vector<T> data(simd_size);
    std::iota(data.begin(), data.end(), 5);
    simd<T> x(data.data(), vector_aligned);
    std::cout << "Reduction algorithms" << std::endl;
    // Reduce default (std::plus)
    T sum_vec = std::accumulate(data.begin(), data.end(), T(0));
    T sum_simd = reduce(x);
    success &= (sum_vec == sum_simd);
    std::cout << "sum: " << sum_vec << " x.reduce(): " << sum_simd << std::endl;
    // Reduce custom
    auto mul_op = [](auto a, auto b){ return a * b; };
    T product_vec = std::accumulate(data.begin(), data.end(), T(1), mul_op);
    T product_simd = reduce(x, mul_op);
    success &= (product_vec == product_simd);
    std::cout << "product: " << product_vec << " x.reduce(mul_op): " << product_simd << std::endl;
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

    std::cout << "\nTesting type: " << "uint8_t" << std::endl;
    success &= test<uint8_t>();
    std::cout << "\nTesting type: " << "uint16_t" << std::endl;
    success &= test<uint16_t>();
    std::cout << "\nTesting type: " << "uint32_t" << std::endl;
    success &= test<uint32_t>();

    std::cout << "\nTesting type: " << "float" << std::endl;
    success &= test<float>();

    return success ? 0 : -1;
}