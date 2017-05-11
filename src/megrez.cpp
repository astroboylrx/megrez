//
// megrez.cpp
//
// Created by Rixin Li on 4/14/17.
// Copyright © 2017 Rixin Li. All rights reserved.
//

// Include the corresponding header file (to ensure it is self-contained)
#include "../include/megrez.hpp"
// Include header files from the same project
#include "../include/megrez_utility.hpp"
// Include header files from non-standard projects
#include "boost/multi_array.hpp"        // Boost MultiArray library
// Include C/C++ libraries
#include <vector>                       // std::vector
#include <numeric>                      // std::iota
#include <algorithm>                    // std::generate
// Include other libraries if needed


namespace megrez {

    /**********************************/
    /********** Library Info **********/
    /**********************************/
    double version() {
        return __version;
    }

    /************************************/
    /********** Array Creation **********/
    /************************************/

    namespace carr {

        double *NativeArr1d(size_t n) {
            double *arr = new double[n];
            std::fill(arr, arr + n, 0.0);

            return arr;
        }

        double **NativeArr2d(size_t n, size_t m) {
            double **arr = new double *[n];
            arr[0] = new double[n * m];
            std::fill(arr[0], arr[0] + n * m, 0.0);

            for (size_t i = 1; i != n; i++) {
                arr[i] = arr[i - 1] + m;
            }

            return arr;
        }

        double ***NativeArr3d(size_t n, size_t m, size_t l) {
            double ***arr = new double **[n];
            arr[0] = new double *[n * m];
            arr[0][0] = new double[n * m * l];
            std::fill(arr[0][0], arr[0][0] + n * m * l, 0.0);

            //arr[0] = &(arr[0][0]); // this is guaranteed by the three new statements
            for (size_t j = 1; j != m; j++) {
                arr[0][j] = arr[0][j - 1] + l;
            }

            for (size_t i = 1; i != n; i++) {
                arr[i] = arr[i - 1] + m;
                arr[i][0] = arr[i - 1][0] + m * l;
                for (size_t j = 1; j != m; j++) {
                    arr[i][j] = arr[i][j - 1] + l;
                }
            }

            return arr;
        }

        double* linspace(double lo, double hi, size_t n) {

            double* arr = NativeArr1d(n);
            if (n <= 0) {
                return arr;
            } else if (n == 1) {
                arr[0] = lo;
            } else {
                auto step = (hi - lo) / (n - 1);
                auto item = lo - step;
                std::generate(arr, arr + n, [&item, &step]() {
                    item += step;
                    return item;
                });
            }

            return arr;
        }

    }

    namespace vec {

        std::vector<size_t> range(size_t n) {
            std::vector<size_t> x(n);
            std::iota(x.begin(), x.end(), 0);
            return x;
        }

    }

    namespace bma {

        boost::multi_array<double, 1> linspace(double lo, double hi, size_t n) {

            boost::multi_array<double, 1> arr;

            if (n <= 0) {
                return arr;
            }

            boost::array<boost::multi_array<double, 1>::index, 1> shape;
            shape[0] = n;
            arr.resize(shape);

            if (n == 1) {
                arr[0] = lo;
            } else {
                auto step = (hi - lo) / (n - 1);
                auto item = lo - step;
                std::generate(arr.data(), arr.data() + n, [&item, &step]() {
                    item += step;
                    return item;
                });
            }

            return arr;
        }

        boost::multi_array<double, 1> zeros(size_t n) {
            boost::multi_array<double, 1> arr;
            arr.resize(boost::extents[n]);
            return arr;
        };

    }
    /**************************************/
    /********** Indices Creation **********/
    /**************************************/

    namespace bma {

        decltype(boost::indices[sn::bma_range()]) Index1d() {
            return boost::indices[sn::bma_range()];
        }

        decltype(boost::indices[sn::bma_range()]) Index1d(size_t lo, size_t hi) {
            return boost::indices[sn::bma_range(lo, hi)];
        }

        decltype(boost::indices[sn::bma_range()]) Index1d(size_t lo, size_t hi, size_t stride) {
            return boost::indices[sn::bma_range(lo, hi, stride)];
        }

        decltype(boost::indices[0][sn::bma_range()]) GetRow(size_t row) {
            return boost::indices[row][sn::bma_range()];
        }

        decltype(boost::indices[0][sn::bma_range()]) GetRow(size_t row, size_t lo, size_t hi) {
            return boost::indices[row][sn::bma_range(lo, hi)];
        }

        decltype(boost::indices[0][sn::bma_range()]) GetRow(size_t row, size_t lo, size_t hi, size_t stride) {
            return boost::indices[row][sn::bma_range(lo, hi, stride)];
        }

        decltype(boost::indices[sn::bma_range()][0]) GetCol(size_t col) {
            return boost::indices[sn::bma_range()][col];
        }

        decltype(boost::indices[sn::bma_range()][0]) GetCol(size_t col, size_t lo, size_t hi) {
            return boost::indices[sn::bma_range(lo, hi)][col];
        }

        decltype(boost::indices[sn::bma_range()][0]) GetCol(size_t col, size_t lo, size_t hi, size_t stride) {
            return boost::indices[sn::bma_range(lo, hi, stride)][col];
        }

        decltype(boost::indices[sn::bma_range()][sn::bma_range()]) Index2d(size_t lo1, size_t hi1, size_t lo2, size_t hi2) {
            return boost::indices[sn::bma_range(lo1, hi1)][sn::bma_range(lo2, hi2)];
        }

        decltype(boost::indices[sn::bma_range()][sn::bma_range()]) Index2d(size_t lo1, size_t hi1, size_t stride1, size_t lo2, size_t hi2, size_t stride2) {
            return boost::indices[sn::bma_range(lo1, hi1, stride1)][sn::bma_range(lo2, hi2, stride2)];
        }

    }
}
