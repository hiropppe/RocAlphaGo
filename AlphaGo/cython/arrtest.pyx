# -*- coding: utf-8 -*-
#cython: boundscheck=False
#cython: wraparound=False

import numpy as np

cimport numpy as np

from libc.stdlib cimport malloc, free

cimport go


cpdef void board_color_speed():
    cdef int[:, ::1] F_np
    cdef int[48][361] F_arr
    cdef int[361] arr
    cdef int i

    go.fill_n_int(arr, 361, 0)
    for i in range(361):
        arr[i] = i%3

    F_np = np.ndarray((48, 361), dtype=np.int32)
    F_np[...] = 0

    for i in range(48):
        go.fill_n_int(F_arr[i], 361, 0)

    import time
    s = time.time()
    for i in range(361):
        if arr[i] == 0:
            F_np[0, i] = 1
        elif arr[i] == 1:
            F_np[1, i] = 1
        else:
            F_np[2, i] = 1

    print 'Stone color(Numpy Memoryview): {:.3f} ns.'.format((time.time() - s) * (1000 ** 3))
    print np.asarray(F_np).reshape((48, 19, 19))

    s = time.time()
    for i in range(361):
        if arr[i] == 0:
            F_arr[0][i] = 1
        elif arr[i] == 1:
            F_arr[1][i] = 1
        else:
            F_arr[2][i] = 1

    print 'Stone color(C Array): {:.3f} ns.'.format((time.time() - s) * (1000 ** 3))
    print np.asarray(F_arr).reshape((48, 19, 19))


cpdef void arr_bench():
    arr_sum()
    arrmv_sum()
    numpy_sum()
    numpymv_sum()


cpdef void arr_sum():
    cdef int i, tmp, sum = 0
    cdef int *arr
    import time
    
    arr = <int *>malloc(100000 * sizeof(int))
    go.fill_n_int(arr, 100000, 0)

    s = time.time()
    for i in range(100000):
        arr[i] = 1
    print 'Array write: ', (time.time() - s) * 1000 ** 2, 'us.'

    s = time.time()
    for i in range(1, 100000):
        tmp = arr[i-1]
    print 'Array read: ', (time.time() - s) * 1000 ** 2, 'us.', tmp

    s = time.time()
    for i in range(100000):
        sum += arr[i]
    print 'Array sum: ', (time.time() - s) * 1000 ** 2, 'us.', sum


cpdef void arrmv_sum():
    cdef int i, tmp, sum = 0
    cdef int *arr
    cdef int[:] mv
    import time
    
    arr = <int *>malloc(100000 * sizeof(int))
    mv = <int[:100000]>arr
    mv[...] = 0

    s = time.time()
    for i in range(100000):
        mv[i] = 1
    print 'Memoryview of c array write: ', (time.time() - s) * 1000 ** 2, 'us.'

    s = time.time()
    for i in range(1, 100000):
        tmp = mv[i-1]
    print 'Memoryview of c array read: ', (time.time() - s) * 1000 ** 2, 'us.', tmp

    s = time.time()
    for i in range(100000):
        sum += mv[i]
    print 'Memoryview of c array sum: ', (time.time() - s) * 1000 ** 2, 'us.', sum


cpdef void numpy_sum():
    cdef int i, tmp, sum = 0
    import time

    arr = np.ndarray((100000), dtype=np.int32)
    arr.fill(0)

    s = time.time()
    for i in range(100000):
        arr[i] = 1 
    print 'ndarray write: ', (time.time() - s) * 1000 ** 2, 'us.'

    s = time.time()
    for i in range(1, 100000):
        tmp = arr[i-1]
    print 'ndarray read: ', (time.time() - s) * 1000 ** 2, 'us.', tmp

    s = time.time()
    sum = arr.sum()
    print 'ndarray sum: ', (time.time() - s) * 1000 ** 2, 'us.', sum


cpdef void numpymv_sum():
    cdef int i, tmp, sum = 0
    cdef int[:] mv
    import time

    arr = np.ndarray((100000), dtype=np.int32)
    arr.fill(0)
    mv = arr

    s = time.time()
    for i in range(100000):
        mv[i] = 1
    print 'Memoryview of ndarray write: ', (time.time() - s) * 1000 ** 2, 'us.'

    s = time.time()
    for i in range(1, 100000):
        tmp = mv[i-1]
    print 'Memoryview of ndarray read: ', (time.time() - s) * 1000 ** 2, 'us.', tmp

    s = time.time()
    for i in range(100000):
        sum += mv[i]
    print 'Memoryview of ndarray sum: ', (time.time() - s) * 1000 ** 2, 'us.', sum
