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
bool test_approximately_equal(T a, T b, T epsilon = 4*std::numeric_limits<T>::epsilon()){
    if constexpr(std::is_floating_point_v<T>)
        return test_true(std::abs(a - b) < epsilon * std::max(std::abs(a), std::abs(b)));
    else
        return test_true(a == b);
}



template <typename T>
bool test_equal(rvv::experimental::simd<T> x, const std::vector<T>& data){

  using printable_type = std::conditional_t<
      std::is_integral_v<T>,
      std::conditional_t<std::is_unsigned_v<T>, uint32_t, int32_t>,
      T>;
  std::cout << "x:    " << x << std::endl;
  std::cout << "data: ( ";
  for (auto el : data) {
    std::cout << printable_type(el) << " ";
    }
    std::cout << ")\n" << std::endl;
    std::vector<T> x_data(x.size());
    x.copy_to(x_data.data(), rvv::experimental::vector_aligned);

    bool success = true;

    for(int i = 0; i < x.size(); i++){
        success &= (x_data[i] == data[i]);
    }

    test_true(success);

    return success;
}

template <typename T>
bool test(){
    bool success = true;
    using namespace rvv::experimental;
    const int simd_size = simd<T>::size();

    std::vector<T> _rand_data(simd_size);
    std::vector<T> _rand_data2(simd_size);
    std::vector<T> _rand_data_positive(simd_size);

    // Fill with random values 
    std::generate(_rand_data.begin(), _rand_data.end(), [](){ return T((std::rand() % 1000)/10.0 - 50); });
    std::generate(_rand_data2.begin(), _rand_data2.end(), [](){ return T( (std::rand() % 1000)/10.0 - 50) ; });
    std::generate(_rand_data_positive.begin(), _rand_data_positive.end(), [](){ return T((std::rand() % 1000)/10.0); });

    const std::vector<T>& rand_data = _rand_data;
    const std::vector<T>& rand_data2 = _rand_data2;
    const std::vector<T>& rand_data_positive = _rand_data_positive;

    {
    simd<T> x;
    std::vector<T> data(rand_data);
    
    std::cout << "Simd copy from" << std::endl;
    x.copy_from(data.data(), vector_aligned);
    success &= test_equal(x, data);

    std::cout << "Simd copy to" << std::endl;
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
        T val = (std::rand() % 100) - 50;
        x.set(i, val);
        success &= test_true(x.get(i) == val);
        success &= test_true(x[i] == val);
        std::cout << "val: " << val << " x.get(i): " << x.get(i) << " x[i]: " << x[i] << std::endl;
    }
    std::cout << std::endl;

    }

    {
    std::vector<T> data(rand_data);
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
    std::for_each(data.begin(), data.end(), [](auto& elem){ elem += 1;});
    success &= test_equal(x, data);
    }

    {
    std::vector<T> data(rand_data);
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
    std::for_each(data.begin(), data.end(), [](auto& elem){ elem -= 1;});
    success &= test_equal(x, data);
    }

    
    {
    std::vector<T> data(rand_data);
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
    std::vector<T> data(rand_data);
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
        std::vector<T> data(rand_data);
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
    std::vector<T> data(rand_data);
    simd<T> x(data.data(), vector_aligned);
    std::cout << "Comparison\n" << std::endl;
    //
    simd<T> y(x);
    success &= test_true((x == y).all_of());
    success &= test_true((x != y).none_of());
    success &= test_true((x < y).none_of());
    success &= test_true((x <= y).all_of());
    success &= test_true((x > y).none_of());
    success &= test_true((x >= y).all_of());
    //
    y = simd<T>(data.data(), vector_aligned);
    y += 1;
    success &= test_true((x == y).none_of());
    success &= test_true((x != y).all_of());
    success &= test_true((x < y).all_of());
    success &= test_true((x <= y).all_of());
    success &= test_true((x > y).none_of());
    success &= test_true((x >= y).none_of());
    }

    // Elementwise Binary Algorithms
    {
    std::vector<T> data_x(rand_data), data_y(rand_data2), data_res(simd_size);


    simd<T> x(data_x.data(), vector_aligned);
    simd<T> y(data_y.data(), vector_aligned);
    simd<T> res(data_res.data(), vector_aligned);

    std::cout << "Elementwise Binary Algorithms" << std::endl;
    // min
    std::cout << "min: " << std::endl;
    std::transform(data_x.begin(), data_x.end(), data_y.begin(), data_res.begin(), [](auto a, auto b){ return std::min(a, b); });
    success &= test_equal(min(x, y), data_res);
    // max
    std::cout << "max: " << std::endl;
    std::transform(data_x.begin(), data_x.end(), data_y.begin(), data_res.begin(), [](auto a, auto b){ return std::max(a, b); });
    success &= test_equal(max(x, y), data_res);
    // copysign
    if constexpr(std::is_floating_point_v<T>)
    {
        std::cout << "copysign: " << std::endl;
        std::transform(data_x.begin(), data_x.end(), data_y.begin(), data_res.begin(), [](auto a, auto b){ return std::copysign(a, b); });
        success &= test_equal(copysign(x, y), data_res);
    }
    }

    // Unary Algorithms
    {

    std::cout << "Unary Algorithms" << std::endl;
    // abs
    if constexpr(std::is_signed_v<T>){
        std::vector<T> data(rand_data);
        simd<T> x(data.data(), vector_aligned);
        std::cout << "abs: " << std::endl;
        std::transform(data.begin(), data.end(), data.begin(), [](const T& a){ return std::abs(a); });
        success &= test_equal(abs(x), data);
    }
    // sqrt
    if constexpr(std::is_floating_point_v<T>)
    {
        std::vector<T> data(rand_data_positive);
        simd<T> x(data.data(), vector_aligned);
        std::cout << "sqrt: " << std::endl;
        std::transform(data.begin(), data.end(), data.begin(), [](auto a){ return std::sqrt(a); });
        success &= test_equal(sqrt(x), data);
    }
    }

    // Reduction algorithms
    {
    std::vector<T> data(rand_data);
    simd<T> x(data.data(), vector_aligned);
    std::cout << "Reduction algorithms" << std::endl;
    // Reduce with default binary operation (std::plus)
    T sum_vec = std::accumulate(data.begin(), data.end(), T(0));
    T sum_simd = reduce(x);
    std::cout << "sum: " << sum_vec << " x.reduce(): " << sum_simd << std::endl;
    success &= (test_approximately_equal(sum_vec, sum_simd));
    // Reduce with custom binary operation
    auto mul_op = [](auto a, auto b){ return a * b; };
    T product_vec = std::accumulate(data.begin(), data.end(), T(1), mul_op);
    T product_simd = reduce(x, mul_op);
    std::cout << "product: " << product_vec << " x.reduce(mul_op): " << product_simd << std::endl;
    success &= (test_approximately_equal(product_vec, product_simd));
    }
    
    return success;
}

int main(){
    bool success = true;

    // seed
    auto seed = std::time(nullptr);
    std::srand(seed);
    std::cout << "seed: " << seed << std::endl;

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