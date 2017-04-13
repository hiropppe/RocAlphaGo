cdef extern from "<random>" namespace "std" nogil:
    # workaround: https://groups.google.com/forum/#!topic/cython-users/xAZxdCFw6Xs
    cdef cppclass mersenne_twister_engine \
        "std::mersenne_twister_engine<class UIntType, size_t w, size_t n, size_t m, size_t r, UIntType a, size_t u, UIntType d, size_t s, UIntType b, size_t t, UIntType c, size_t l, UIntType f>"[UIntType]:
        mersenne_twister_engine(UIntType)
    ctypedef mersenne_twister_engine[int] mt19937 "std::mersenne_twister_engine<uint_fast32_t, 32, 624, 397, 31, 0x9908b0df, 11, 0xffffffff, 7, 0x9d2c5680, 15, 0xefc60000, 18, 1812433253>"
    ctypedef mersenne_twister_engine[long long] mt19937_64 "std::mersenne_twister_engine<uint_fast64_t, 64, 312, 156, 31, 0xb5026f5aa96619e9, 29, 0x5555555555555555,17, 0x71d67fffeda60000, 37, 0xfff7eee000000000, 43, 6364136223846793005>"

    cdef cppclass random_device:
        random_device()
        unsigned int operator()()

    cdef cppclass uniform_int_distribution[T]:
        uniform_int_distribution(T, T) except +
        T operator()[U](U&);

#cdef extern from "<algorithm>" namespace "std":
#    void std_sort "std:sort" [iter](iter first, iter last)
