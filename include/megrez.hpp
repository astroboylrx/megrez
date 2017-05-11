//
// megrez.hpp
//
// Created by Rixin Li on 4/14/17.
// Copyright © 2017 Rixin Li. All rights reserved.
//

#ifndef MEGREZ_LIBRARY_H
#define MEGREZ_LIBRARY_H

// Include header files from the same project
#include "megrez_utility.hpp"           // megrez::PromoteNumeric
// Include header files from non-standard projects
#include "boost/multi_array.hpp"        // Boost MultiArray library
// Include C/C++ libraries
#include <cmath>                        // std::pow, std::sqrt
#include <iostream>                     // std::cout
#include <vector>                       // std::vector
#include <numeric>                      // std::iota
#include <algorithm>                    // std::generate, std::transform, std::fill
// Include other libraries if needed


namespace megrez {

    /**********************************/
    /********** Library Info **********/
    /**********************************/

    const double __version = 1.0;

    double version();

    /******************************************/
    /********** Supporting Utilities **********/
    /******************************************/

    namespace bma {

        template<size_t N>
        bool is_same_shape(const size_t *lhs, const size_t *rhs, bool quiet = false) {
            for (unsigned int i = 0; i != N; i++) {
                if (lhs[i] != rhs[i]) {
                    if (!quiet) {
                        std::cerr << "Error: operations must be done between two multi_arries with the same shape."
                                  << std::endl;
                    }
                    return false;
                }
            }
            return true;
        }

    }

    /************************************/
    /********** Array Creation **********/
    /************************************/

    namespace carr {

        template<class T>
        T *NativeArr1d(size_t n) {
            T *arr = new T[n];
            std::fill(arr, arr + n, static_cast<T>(0));
            return arr;
        }

        double *NativeArr1d(size_t n);

        template<class T>
        void DeleteNativeArr1d(T *arr) {
            if (arr != nullptr) {
                delete[] arr;
            }
            arr = nullptr;
        }

        template<class T>
        T **NativeArr2d(size_t n, size_t m) {
            T **arr = new T *[n];
            arr[0] = new T[n * m];
            std::fill(arr[0], arr[0] + n * m, static_cast<T>(0));

            for (size_t i = 1; i != n; i++) {
                arr[i] = arr[i - 1] + m;
            }
            return arr;
        }

        double **NativeArr2d(size_t n, size_t m);

        template<class T>
        void DeleteNativeArr2d(T **arr) {
            if (arr != nullptr) {
                delete[] arr[0];
                arr[0] = nullptr;
            }
            delete[] arr;
            arr = nullptr;
        }

        template<class T>
        T ***NativeArr3d(size_t n, size_t m, size_t l) {
            T ***arr = new T **[n];
            arr[0] = new T *[n * m];
            arr[0][0] = new T[n * m * l];
            std::fill(arr[0][0], arr[0][0] + n * m * l, static_cast<T>(0));

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

        double ***NativeArr3d(size_t n, size_t m, size_t l);

        template<class T>
        void DeleteNativeArr3d(T ***arr) {
            if (arr != nullptr) {
                delete[] arr[0][0];
                arr[0][0] = nullptr;
                delete[] arr[0];
                arr[0] = nullptr;
            }
            delete[] arr;
            arr = nullptr;
        }

        template<class T, class U>
        T* linspace(U lo, U hi, size_t n) {
            T* arr = NativeArr1d<T>(n);
            if (n <= 0) {
                return arr;
            } else if (n == 1) {
                arr[0] = lo;
            } else {
                auto step = double(hi - lo) / (n - 1);
                auto item = lo - step;
                std::generate(arr, arr + n, [&item, &step]() {
                    item += step;
                    return static_cast<T>(item);
                });
            }
            return arr;
        };

        double* linspace(double lo, double hi, size_t n);


    }

    namespace vec {

        template<typename T>
        std::vector<T> range(size_t n) {
            std::vector<T> x(n);
            std::iota(x.begin(), x.end(), 0);
            return x;
        }

        template<typename T>
        std::vector<T> range(T lo, T hi) {
            std::vector<T> x;
            x.resize(static_cast<size_t>(std::fabs(hi - lo)));
            T step = std::copysign(1, hi - lo);
            auto item = lo - step;
            std::generate(x.begin(), x.end(), [&item, &step]() {
                item += step;
                return item;
            });
            return x;
        }

        template<typename T>
        std::vector<T> range(T lo, T hi, T step) {
            if (step == 0) {
                std::cerr << "Error in range(): step should not be zero (changed to 1)." << std::endl;
                step = 1;
            }
            std::vector<T> x;
            x.resize(static_cast<size_t>((hi - lo) / step));
            auto item = lo - step;
            std::generate(x.begin(), x.end(), [&item, &step]() {
                item += step;
                return item;
            });
            return x;
        }

        std::vector<size_t> range(size_t n);

    }

    namespace bma {

        template<class T, class U>
        boost::multi_array<T, 1> linspace(U lo, U hi, size_t n) {
            boost::multi_array<T, 1> arr;

            if (n <= 0) {
                return arr;
            }

            typename boost::array<typename boost::multi_array<T, 1>::index, 1> shape;
            shape[0] = n;
            arr.resize(shape);

            if (n == 1) {
                arr[0] = lo;
            } else {
                auto step = double(hi - lo) / (n - 1);
                auto item = lo - step;
                std::generate(arr.data(), arr.data() + n, [&item, &step]() {
                    item += step;
                    return item;
                });
            }
            return arr;
        };

        boost::multi_array<double, 1> linspace(double lo, double hi, size_t n);

        template<class T, size_t N, class... Shape>
        boost::multi_array<T, N>
        zeros(typename std::enable_if<sizeof...(Shape) + 1 == N, size_t>::type head, Shape... shape) {
            size_t *s = new size_t[N]{head, size_t(shape)...};
            boost::multi_array<T, N> arr;
            arr.resize(reinterpret_cast<const boost::array<size_t, N> &>(*s));
            delete[] s;

            return arr;
        };

        boost::multi_array<double, 1> zeros(size_t n);

        template<class T, size_t N>
        boost::multi_array<T, N> zeros_like(const boost::multi_array<T, N> &model) {
            boost::multi_array<T, N> arr;
            arr.resize(reinterpret_cast<const boost::array<size_t, N> &>(*model.shape()));

            return arr;
        };

        template<class T, class U, size_t N>
        boost::multi_array<T, N> zeros_like(const boost::multi_array<U, N> &model) {
            boost::multi_array<T, N> arr;
            arr.resize(reinterpret_cast<const boost::array<size_t, N> &>(*model.shape()));

            return arr;
        };

        template<class T, class U>
        typename std::enable_if<(T::dimensionality == U::dimensionality), void>::type BmaAssign(T &lhs, const U &rhs) {
            if (!is_same_shape<T::dimensionality>(lhs.shape(), rhs.shape(), true)) {
                lhs.resize(reinterpret_cast<const boost::array<size_t, T::dimensionality> &>(*rhs.shape()));
            }
            lhs = rhs;
        }

    }

    /************************************/
    /********** Views Creation **********/
    /************************************/

    namespace bma {

        /*! \functor template <typename RangeArrayType, size_t Dimension> IndicesBuilder
         *  \brief a functor to build indices */
        template<typename RangeArrayType, size_t Dimension>
        struct IndicesBuilder {
            // Recursively invoke the functor for the next lowest dimension and add the next range.
            static auto
            Build(const RangeArrayType &range) -> decltype(IndicesBuilder<RangeArrayType, Dimension - 1>::Build(
                    range)[range[Dimension - 1]]) {
                return IndicesBuilder<RangeArrayType, Dimension - 1>::Build(range)[range[Dimension - 1]];
            }
        };

        /*! \functor template <typename RangeArrayType> IndicesBuilder<RangeArrayType, 1>
         *  \brief a functor specialization to terminate recursion when Dimension=1 */
        template<typename RangeArrayType>
        struct IndicesBuilder<RangeArrayType, 1> {
            /*
             * In C++11, there are two syntax for function declaration:
             *     return-type identifier ( argument-declarations... )
             * and
             *     auto identifier ( argument-declarations... ) -> return_type
             * They are equivalent. But with the later one, you can specify the return_type using decltype(...), where ... is/are only declared in the argument-declarations. For example:
             *     template <typename T1, typename T2>
             *     auto compose(T1 a, T2 b) -> decltype(a + b);
             */
            static auto Build(const RangeArrayType &range) -> decltype(boost::indices[range[0]]) {
                return boost::indices[range[0]];
            }
        };

        /*! \fn template <typename U, size_t Dimension> typename boost::multi_array<U, Dimension>::template array_view<Dimension>::type ExtractSubArrayView(boost::multi_array<U, Dimension>& array, const boost::array<size_t, Dimension>& corner, const boost::array<size_t, Dimension>& subarray_size)
         *  \brief function to extract a view of subarray (corner, subarray_size) from the master array */
        template<typename U, size_t Dimension>
        typename boost::multi_array<U, Dimension>::template array_view<Dimension>::type
        ExtractSubArrayView(boost::multi_array<U, Dimension> &array, const boost::array<size_t, Dimension> &corner,
                            const boost::array<size_t, Dimension> &subarray_size) {

            using array_type = boost::multi_array<U, Dimension>;
            using range_type = typename array_type::index_range;

            // Build a random-access container with the ranges.
            std::vector<range_type> range;
            for (size_t i = 0; i != Dimension; ++i) {
                range.push_back(range_type(corner[i], corner[i] + subarray_size[i]));
            }

            // Use the helper functor to build the index object.
            auto index = IndicesBuilder<decltype(range), Dimension>::Build(range);

            typename array_type::template array_view<Dimension>::type view = array[index];
            return view;
        }

    }
    /**************************************/
    /********** Indices Creation **********/
    /**************************************/

    namespace bma {

        decltype(boost::indices[sn::bma_range()]) Index1d(size_t lo, size_t hi);

        decltype(boost::indices[sn::bma_range()]) Index1d(size_t lo, size_t hi, size_t stride);

        decltype(boost::indices[0][sn::bma_range()]) GetRow(size_t row);

        decltype(boost::indices[0][sn::bma_range()]) GetRow(size_t row, size_t lo, size_t hi);

        decltype(boost::indices[0][sn::bma_range()]) GetRow(size_t row, size_t lo, size_t hi, size_t stride);

        decltype(boost::indices[sn::bma_range()][0]) GetCol(size_t col);

        decltype(boost::indices[sn::bma_range()][0]) GetCol(size_t col, size_t lo, size_t hi);

        decltype(boost::indices[sn::bma_range()][0]) GetCol(size_t col, size_t lo, size_t hi, size_t stride);

        decltype(boost::indices[sn::bma_range()][sn::bma_range()]) Index2d(size_t lo1, size_t hi1, size_t lo2, size_t hi2);

        decltype(boost::indices[sn::bma_range()][sn::bma_range()]) Index2d(size_t lo1, size_t hi1, size_t stride1, size_t lo2, size_t hi2, size_t stride2);

    }

    /**************************************/
    /********** Array Operations **********/
    /**************************************/

    namespace bma {

        /*! \fn template <class U, class F> typename std::enable_if<(U::dimensionality==1), void>::type IterateArrayView(U& array, F f)
         *  \brief function specialization to iterate the final dimension (f is specialized to accept one argument) */
        template<class U, class F>
        typename std::enable_if<(U::dimensionality == 1), void>::type IterateBoostMultiArrayConcept(U &array, F f) {
            for (auto &element : array) {
                f(element);
            }
        }

        /*! \fn template <class U, class F> typename std::enable_if<(U::dimensionality>1), void>::type IterateBoostMultiArrayConcept(U& array, F f)
         *  \brief function to iterate over an Boost MultiArray concept object (f is specialized to accept one argument) */
        template<class U, class F>
        typename std::enable_if<(U::dimensionality > 1), void>::type IterateBoostMultiArrayConcept(U &array, F f) {
            for (auto element : array) {
                IterateBoostMultiArrayConcept<decltype(element), F>(element, f);
            }
        }

        template<class T, class U, class F>
        typename std::enable_if<(T::dimensionality == 1 && U::dimensionality == 1), void>::type
        ParaIterateBoostMultiArrayConcept(const T &arr1, const U &arr2, F f) {
            auto item2 = arr2.begin();
            for (auto &item1 : arr1) {
                f(item1, *item2);
                item2++;
            }
        };

        template<class T, class U, class F>
        typename std::enable_if<(T::dimensionality == U::dimensionality && T::dimensionality > 1), void>::type
        ParaIterateBoostMultiArrayConcept(const T &arr1, const U &arr2, F f) {
            auto item2 = arr2.begin();
            for (auto item1 : arr1) {
                ParaIterateBoostMultiArrayConcept<typename T::template const_subarray<
                        T::dimensionality - 1>::type, typename U::template const_subarray<
                        U::dimensionality - 1>::type, F>(item1, *item2, f);
                item2++;
            }
        };

        template<class T, class Op>
        boost::multi_array<typename T::element, T::dimensionality> SerialUnaryOperation(const T &lhs, Op op) {
            boost::multi_array<typename T::element, T::dimensionality> tmp;
            tmp.resize(reinterpret_cast<const boost::array<size_t, T::dimensionality> &>(*lhs.shape()));
            auto item = tmp.data();
            IterateBoostMultiArrayConcept(lhs, [&item, &op](const typename T::element &item1) {
                *item = op(item1);
                item++;
            });

            return tmp;
        }

        template<class T, class U, class Op>
        boost::multi_array<typename PromoteNumeric<typename T::element, U>::type, T::dimensionality>
        SerialBinaryOperation(const T &lhs, const U &rhs, Op op) {
            boost::multi_array<typename PromoteNumeric<typename T::element, U>::type, T::dimensionality> tmp;
            tmp.resize(reinterpret_cast<const boost::array<size_t, T::dimensionality> &>(*lhs.shape()));
            auto item = tmp.data();
            IterateBoostMultiArrayConcept(lhs, [&item, &rhs, &op](const typename T::element &item1) {
                *item = op(item1, rhs);
                item++;
            });

            return tmp;
        }

        template<class T, class U, class Op, typename std::enable_if<(T::dimensionality ==
                                                                      U::dimensionality), int>::type= 0>
        boost::multi_array<typename PromoteNumeric<typename T::element, typename U::element>::type, T::dimensionality>
        ParaOperation(const T &lhs, const U &rhs, Op op) {
            boost::multi_array<typename PromoteNumeric<typename T::element, typename U::element>::type, T::dimensionality> tmp;
            if (!is_same_shape<T::dimensionality>(lhs.shape(), rhs.shape())) {
                return tmp; // intentionally return it before resizing it to cause errors
            }
            tmp.resize(reinterpret_cast<const boost::array<size_t, T::dimensionality> &>(*lhs.shape()));
            auto item = tmp.data();
            ParaIterateBoostMultiArrayConcept(lhs, rhs, [&item, &op](const typename T::element &item1,
                                                                     const typename U::element &item2) {
                *item = op(item1, item2);
                item++;
            });

            return tmp;
        }

        namespace boost_multi_array_concept_operators {

            /**************************************/
            /********** Array and Scalar **********/
            /**************************************/

            template<class T, class U, typename std::enable_if<(std::is_arithmetic<U>::value), int>::type= 0>
            inline boost::multi_array<typename PromoteNumeric<typename T::element, U>::type, T::dimensionality>
            operator+(const T &lhs, const U &rhs) {
                return SerialBinaryOperation(lhs, rhs,
                                             std::plus<typename PromoteNumeric<typename T::element, U>::type>());
            };

            template<class T, class U, typename std::enable_if<(std::is_arithmetic<T>::value), int>::type= 0>
            inline boost::multi_array<typename PromoteNumeric<T, typename U::element>::type, U::dimensionality>
            operator+(const T &lhs, const U &rhs) {
                return SerialBinaryOperation(rhs, lhs,
                                             std::plus<typename PromoteNumeric<T, typename U::element>::type>());
            };

            template<class T, class U, typename std::enable_if<(std::is_arithmetic<U>::value), int>::type= 0>
            inline boost::multi_array<typename PromoteNumeric<typename T::element, U>::type, T::dimensionality>
            operator-(const T &lhs, const U &rhs) {
                return SerialBinaryOperation(lhs, rhs,
                                             std::minus<typename PromoteNumeric<typename T::element, U>::type>());
            };

            template<class T, class U, typename std::enable_if<(std::is_arithmetic<T>::value), int>::type= 0>
            inline boost::multi_array<typename PromoteNumeric<T, typename U::element>::type, U::dimensionality>
            operator-(const T &lhs, const U &rhs) {
                return SerialBinaryOperation(rhs, lhs, [](const typename U::element &y, const T &x) {
                    return x - y;
                });
            };

            template<class T, class U, typename std::enable_if<(std::is_arithmetic<U>::value), int>::type= 0>
            inline boost::multi_array<typename PromoteNumeric<typename T::element, U>::type, T::dimensionality>
            operator*(const T &lhs, const U &rhs) {
                return SerialBinaryOperation(lhs, rhs,
                                             std::multiplies<typename PromoteNumeric<typename T::element, U>::type>());
            };

            template<class T, class U, typename std::enable_if<(std::is_arithmetic<T>::value), int>::type= 0>
            inline boost::multi_array<typename PromoteNumeric<T, typename U::element>::type, U::dimensionality>
            operator*(const T &lhs, const U &rhs) {
                return SerialBinaryOperation(rhs, lhs,
                                             std::multiplies<typename PromoteNumeric<T, typename U::element>::type>());
            };

            template<class T, class U, typename std::enable_if<(std::is_arithmetic<U>::value), int>::type= 0>
            inline boost::multi_array<typename PromoteNumeric<typename T::element, U>::type, T::dimensionality>
            operator/(const T &lhs, const U &rhs) {
                return SerialBinaryOperation(lhs, rhs,
                                             std::divides<typename PromoteNumeric<typename T::element, U>::type>());
            };

            template<class T, class U, typename std::enable_if<(std::is_arithmetic<T>::value), int>::type= 0>
            inline boost::multi_array<typename PromoteNumeric<T, typename U::element>::type, U::dimensionality>
            operator/(const T &lhs, const U &rhs) {
                return SerialBinaryOperation(rhs, lhs, [](const typename U::element &y, const T &x) {
                    return x / y;
                });
            }

            /**********************************************/
            /********** Array with the same size **********/
            /**********************************************/

            template<class T, class U, typename std::enable_if<(T::dimensionality == U::dimensionality), int>::type= 0>
            inline boost::multi_array<typename PromoteNumeric<typename T::element, typename U::element>::type, T::dimensionality>
            operator+(const T &lhs, const U &rhs) {
                return ParaOperation(lhs, rhs,
                                     std::plus<typename PromoteNumeric<typename T::element, typename U::element>::type>());
            };

            template<class T, class U, typename std::enable_if<(T::dimensionality == U::dimensionality), int>::type= 0>
            inline boost::multi_array<typename PromoteNumeric<typename T::element, typename U::element>::type, T::dimensionality>
            operator-(const T &lhs, const U &rhs) {
                return ParaOperation(lhs, rhs,
                                     std::minus<typename PromoteNumeric<typename T::element, typename U::element>::type>());
            };

            template<class T, class U, typename std::enable_if<(T::dimensionality == U::dimensionality), int>::type= 0>
            inline boost::multi_array<typename PromoteNumeric<typename T::element, typename U::element>::type, T::dimensionality>
            operator*(const T &lhs, const U &rhs) {
                return ParaOperation(lhs, rhs,
                                     std::multiplies<typename PromoteNumeric<typename T::element, typename U::element>::type>());
            };

            template<class T, class U, typename std::enable_if<(T::dimensionality == U::dimensionality), int>::type= 0>
            inline boost::multi_array<typename PromoteNumeric<typename T::element, typename U::element>::type, T::dimensionality>
            operator/(const T &lhs, const U &rhs) {
                return ParaOperation(lhs, rhs,
                                     std::divides<typename PromoteNumeric<typename T::element, typename U::element>::type>());
            };

            /****************************************/
            /********** Array broadcasting **********/
            /****************************************/

            /*
            template <class T, class U, typename std::enable_if<(T::dimensionality > U::dimensionality), int>::type=0>
            inline boost::multi_array<typename PromoteNumeric<typename T::element, typename U::element>::type, T::dimensionality> operator + (const T& lhs, const U& rhs) {
                boost::multi_array<typename PromoteNumeric<typename T::element, typename U::element>::type, T::dimensionality> tmp;
                if (is_same_shape<U::dimensionality>(lhs.shape()+T::dimensionality-U::dimensionality, rhs.shape())) { // last few dimentions in the same shape
                    tmp.resize(reinterpret_cast<const boost::array<size_t, U::dimensionality>&>(*lhs.shape()));
                    auto item = tmp.begin();
                    for (auto sub_item : lhs) {
                        *item = sub_item + lhs; // all operations need to be valid for this template function
                        item++;
                    }
                } else if (is_same_shape<U::dimensionality>(lhs.shape(), rhs.shape())) { // first few dimensions in the same shape
                    tmp.resize(reinterpret_cast<const boost::array<size_t, U::dimensionality>&>(*lhs.shape()));
                    typename T::index_gen range;
                    for (size_t i = 0; i != T::dimensionality-1; i++) {
                        range = range[sn::bma_range()];
                    }

                    for (size_t i = 0; i != lhs.shape()[T::dimensionality-1]; i++) {
                        tmp[range[i]] = lhs[range[i]] + rhs; // all operations need to be valid for this template function
                    }
                } else {
                    std::cerr << "Additional error information: this operation involves broadcasting array." << std::endl;
                }

                return tmp;
            };
             */

            template<class T, class U, typename std::enable_if<(T::dimensionality < U::dimensionality), int>::type= 0>
            inline boost::multi_array<typename PromoteNumeric<typename T::element, typename U::element>::type, U::dimensionality>
            operator+(const T &lhs, const U &rhs) {
                boost::multi_array<typename PromoteNumeric<typename T::element, typename U::element>::type, U::dimensionality> tmp;
                if (!is_same_shape<T::dimensionality>(lhs.shape(),
                                                      rhs.shape() + U::dimensionality - T::dimensionality)) {
                    std::cerr << "Additional error information: this operation involves broadcasting array."
                              << std::endl;
                    return tmp; // intentionally return it before resizing it to cause errors
                }
                tmp.resize(reinterpret_cast<const boost::array<size_t, U::dimensionality> &>(*rhs.shape()));
                auto item = tmp.begin();
                for (auto sub_item : rhs) {
                    *item = sub_item + lhs;
                    item++;
                }

                return tmp;
            };

            template<class T, class U, typename std::enable_if<(T::dimensionality > U::dimensionality), int>::type= 0>
            inline boost::multi_array<typename PromoteNumeric<typename T::element, typename U::element>::type, T::dimensionality>
            operator+(const T &lhs, const U &rhs) {
                return rhs + lhs;
            };

            template<class T, class U, typename std::enable_if<(T::dimensionality < U::dimensionality), int>::type= 0>
            inline boost::multi_array<typename PromoteNumeric<typename T::element, typename U::element>::type, U::dimensionality>
            operator-(const T &lhs, const U &rhs) {
                boost::multi_array<typename PromoteNumeric<typename T::element, typename U::element>::type, U::dimensionality> tmp;
                if (!is_same_shape<T::dimensionality>(lhs.shape(),
                                                      rhs.shape() + U::dimensionality - T::dimensionality)) {
                    std::cerr << "Additional error information: this operation involves broadcasting array."
                              << std::endl;
                    return tmp; // intentionally return it before resizing it to cause errors
                }
                tmp.resize(reinterpret_cast<const boost::array<size_t, U::dimensionality> &>(*rhs.shape()));
                auto item = tmp.begin();
                for (auto sub_item : rhs) {
                    *item = lhs - sub_item;
                    item++;
                }

                return tmp;
            };

            template<class T, class U, typename std::enable_if<(T::dimensionality > U::dimensionality), int>::type= 0>
            inline boost::multi_array<typename PromoteNumeric<typename T::element, typename U::element>::type, T::dimensionality>
            operator-(const T &lhs, const U &rhs) {
                boost::multi_array<typename PromoteNumeric<typename T::element, typename U::element>::type, T::dimensionality> tmp;
                if (!is_same_shape<U::dimensionality>(lhs.shape() + T::dimensionality - U::dimensionality,
                                                      rhs.shape())) {
                    std::cerr << "Additional error information: this operation involves broadcasting array."
                              << std::endl;
                    return tmp; // intentionally return it before resizing it to cause errors
                }
                tmp.resize(reinterpret_cast<const boost::array<size_t, T::dimensionality> &>(*lhs.shape()));
                auto item = tmp.begin();
                for (auto sub_item : lhs) {
                    *item = sub_item - rhs;
                    item++;
                }

                return tmp;
            };

            template<class T, class U, typename std::enable_if<(T::dimensionality < U::dimensionality), int>::type= 0>
            inline boost::multi_array<typename PromoteNumeric<typename T::element, typename U::element>::type, U::dimensionality>
            operator*(const T &lhs, const U &rhs) {
                boost::multi_array<typename PromoteNumeric<typename T::element, typename U::element>::type, U::dimensionality> tmp;
                if (!is_same_shape<T::dimensionality>(lhs.shape(),
                                                      rhs.shape() + U::dimensionality - T::dimensionality)) {
                    std::cerr << "Additional error information: this operation involves broadcasting array."
                              << std::endl;
                    return tmp; // intentionally return it before resizing it to cause errors
                }
                tmp.resize(reinterpret_cast<const boost::array<size_t, U::dimensionality> &>(*rhs.shape()));
                auto item = tmp.begin();
                for (auto sub_item : rhs) {
                    *item = sub_item * lhs;
                    item++;
                }

                return tmp;
            };

            template<class T, class U, typename std::enable_if<(T::dimensionality > U::dimensionality), int>::type= 0>
            inline boost::multi_array<typename PromoteNumeric<typename T::element, typename U::element>::type, T::dimensionality>
            operator*(const T &lhs, const U &rhs) {
                return rhs * lhs;
            };

            template<class T, class U, typename std::enable_if<(T::dimensionality < U::dimensionality), int>::type= 0>
            inline boost::multi_array<typename PromoteNumeric<typename T::element, typename U::element>::type, U::dimensionality>
            operator/(const T &lhs, const U &rhs) {
                boost::multi_array<typename PromoteNumeric<typename T::element, typename U::element>::type, U::dimensionality> tmp;
                if (!is_same_shape<T::dimensionality>(lhs.shape(),
                                                      rhs.shape() + U::dimensionality - T::dimensionality)) {
                    std::cerr << "Additional error information: this operation involves broadcasting array."
                              << std::endl;
                    return tmp; // intentionally return it before resizing it to cause errors
                }
                tmp.resize(reinterpret_cast<const boost::array<size_t, U::dimensionality> &>(*rhs.shape()));
                auto item = tmp.begin();
                for (auto sub_item : rhs) {
                    *item = lhs / sub_item;
                    item++;
                }

                return tmp;
            };

            template<class T, class U, typename std::enable_if<(T::dimensionality > U::dimensionality), int>::type= 0>
            inline boost::multi_array<typename PromoteNumeric<typename T::element, typename U::element>::type, T::dimensionality>
            operator/(const T &lhs, const U &rhs) {
                boost::multi_array<typename PromoteNumeric<typename T::element, typename U::element>::type, T::dimensionality> tmp;
                if (!is_same_shape<U::dimensionality>(lhs.shape() + T::dimensionality - U::dimensionality,
                                                      rhs.shape())) {
                    std::cerr << "Additional error information: this operation involves broadcasting array."
                              << std::endl;
                    return tmp; // intentionally return it before resizing it to cause errors
                }
                tmp.resize(reinterpret_cast<const boost::array<size_t, T::dimensionality> &>(*lhs.shape()));
                auto item = tmp.begin();
                for (auto sub_item : lhs) {
                    *item = sub_item / rhs;
                    item++;
                }

                return tmp;
            };

        }

        namespace boost_multi_array_operators {

            template<class T, class U, size_t N>
            inline boost::multi_array<typename PromoteNumeric<T, U>::type, N>
            operator+(const boost::multi_array<T, N> &lhs, const boost::multi_array<U, N> &rhs) {
                boost::multi_array<typename PromoteNumeric<T, U>::type, N> tmp;
                if (!is_same_shape<N>(lhs.shape(), rhs.shape())) {
                    return tmp;
                }
                tmp.resize(reinterpret_cast<const boost::array<size_t, N> &>(*lhs.shape()));
                std::transform(lhs.data(), lhs.data() + lhs.num_elements(), rhs.data(), tmp.data(),
                               std::plus<typename PromoteNumeric<T, U>::type>());

                return tmp;
            }

            template<class T, class U, size_t N>
            inline boost::multi_array<typename PromoteNumeric<T, U>::type, N>
            operator+(const boost::multi_array<T, N> &lhs, const U &rhs) {
                boost::multi_array<typename PromoteNumeric<T, U>::type, N> tmp;
                tmp.resize(reinterpret_cast<const boost::array<size_t, N> &>(*lhs.shape()));
                std::transform(lhs.data(), lhs.data() + lhs.num_elements(), tmp.data(), [&rhs](T item) {
                    return item + rhs;
                });

                return tmp;
            }

            template<class T, class U, size_t N>
            inline boost::multi_array<typename PromoteNumeric<T, U>::type, N>
            operator+(const T &lhs, const boost::multi_array<U, N> &rhs) {
                boost::multi_array<typename PromoteNumeric<T, U>::type, N> tmp;
                tmp.resize(reinterpret_cast<const boost::array<size_t, N> &>(*rhs.shape()));
                std::transform(rhs.data(), rhs.data() + rhs.num_elements(), tmp.data(), [&lhs](U item) {
                    return lhs + item;
                });

                return tmp;
            }

            template<class T, class U, size_t N>
            inline boost::multi_array<typename PromoteNumeric<T, U>::type, N>
            operator-(const boost::multi_array<T, N> &lhs, const boost::multi_array<U, N> &rhs) {
                boost::multi_array<typename PromoteNumeric<T, U>::type, N> tmp;
                if (!is_same_shape<N>(lhs.shape(), rhs.shape())) {
                    return tmp;
                }
                tmp.resize(reinterpret_cast<const boost::array<size_t, N> &>(*lhs.shape()));
                std::transform(lhs.data(), lhs.data() + lhs.num_elements(), rhs.data(), tmp.data(),
                               std::minus<typename PromoteNumeric<T, U>::type>());

                return tmp;
            }

            template<class T, class U, size_t N>
            inline boost::multi_array<typename PromoteNumeric<T, U>::type, N>
            operator-(const boost::multi_array<T, N> &lhs, const U &rhs) {
                boost::multi_array<typename PromoteNumeric<T, U>::type, N> tmp;
                tmp.resize(reinterpret_cast<const boost::array<size_t, N> &>(*lhs.shape()));
                std::transform(lhs.data(), lhs.data() + lhs.num_elements(), tmp.data(), [&rhs](T item) {
                    return item - rhs;
                });

                return tmp;
            }

            template<class T, class U, size_t N>
            inline boost::multi_array<typename PromoteNumeric<T, U>::type, N>
            operator-(const T &lhs, const boost::multi_array<U, N> &rhs) {
                boost::multi_array<typename PromoteNumeric<T, U>::type, N> tmp;
                tmp.resize(reinterpret_cast<const boost::array<size_t, N> &>(*rhs.shape()));
                std::transform(rhs.data(), rhs.data() + rhs.num_elements(), tmp.data(), [&lhs](U item) {
                    return lhs - item;
                });

                return tmp;
            }

            template<class T, class U, size_t N>
            inline boost::multi_array<typename PromoteNumeric<T, U>::type, N>
            operator*(const boost::multi_array<T, N> &lhs, const boost::multi_array<U, N> &rhs) {
                boost::multi_array<typename PromoteNumeric<T, U>::type, N> tmp;
                if (!is_same_shape<N>(lhs.shape(), rhs.shape())) {
                    return tmp;
                }
                tmp.resize(reinterpret_cast<const boost::array<size_t, N> &>(*lhs.shape()));
                std::transform(lhs.data(), lhs.data() + lhs.num_elements(), rhs.data(), tmp.data(),
                               std::multiplies<typename PromoteNumeric<T, U>::type>());

                return tmp;
            }

            template<class T, class U, size_t N>
            inline boost::multi_array<typename PromoteNumeric<T, U>::type, N>
            operator*(const boost::multi_array<T, N> &lhs, const U &rhs) {
                boost::multi_array<typename PromoteNumeric<T, U>::type, N> tmp;
                tmp.resize(reinterpret_cast<const boost::array<size_t, N> &>(*lhs.shape()));
                std::transform(lhs.data(), lhs.data() + lhs.num_elements(), tmp.data(), [&rhs](T item) {
                    return item * rhs;
                });

                return tmp;
            }

            template<class T, class U, size_t N>
            inline boost::multi_array<typename PromoteNumeric<T, U>::type, N>
            operator*(const T &lhs, const boost::multi_array<U, N> &rhs) {
                boost::multi_array<typename PromoteNumeric<T, U>::type, N> tmp;
                tmp.resize(reinterpret_cast<const boost::array<size_t, N> &>(*rhs.shape()));
                std::transform(rhs.data(), rhs.data() + rhs.num_elements(), tmp.data(), [&lhs](U item) {
                    return lhs * item;
                });

                return tmp;
            }

            template<class T, class U, size_t N>
            inline boost::multi_array<typename PromoteNumeric<T, U>::type, N>
            operator/(const boost::multi_array<T, N> &lhs, const boost::multi_array<U, N> &rhs) {
                boost::multi_array<typename PromoteNumeric<T, U>::type, N> tmp;
                if (!is_same_shape<N>(lhs.shape(), rhs.shape())) {
                    return tmp;
                }
                tmp.resize(reinterpret_cast<const boost::array<size_t, N> &>(*lhs.shape()));
                std::transform(lhs.data(), lhs.data() + lhs.num_elements(), rhs.data(), tmp.data(),
                               std::divides<typename PromoteNumeric<T, U>::type>());

                return tmp;
            }

            template<class T, class U, size_t N>
            inline boost::multi_array<typename PromoteNumeric<T, U>::type, N>
            operator/(const boost::multi_array<T, N> &lhs, const U &rhs) {
                boost::multi_array<typename PromoteNumeric<T, U>::type, N> tmp;
                tmp.resize(reinterpret_cast<const boost::array<size_t, N> &>(*lhs.shape()));
                std::transform(lhs.data(), lhs.data() + lhs.num_elements(), tmp.data(), [&rhs](T item) {
                    return item / rhs;
                });

                return tmp;
            }

            template<class T, class U, size_t N>
            inline boost::multi_array<typename PromoteNumeric<T, U>::type, N>
            operator/(const T &lhs, const boost::multi_array<U, N> &rhs) {
                boost::multi_array<typename PromoteNumeric<T, U>::type, N> tmp;
                tmp.resize(reinterpret_cast<const boost::array<size_t, N> &>(*rhs.shape()));
                std::transform(rhs.data(), rhs.data() + rhs.num_elements(), tmp.data(), [&lhs](U item) {
                    return lhs / item;
                });

                return tmp;
            }

        }

        template<class T, class U>
        inline boost::multi_array<typename PromoteNumeric<typename T::element, U>::type, T::dimensionality>
        pow(const T &arr, const U &power) {
            return SerialBinaryOperation(arr, power, [](const typename T::element &element, const U &power) {
                return std::pow(element, power);
            });
            /*
            boost::multi_array<typename PromoteNumeric<typename T::element, U>::type, T::dimensionality> tmp;
            tmp.resize(reinterpret_cast<const boost::array<size_t, T::dimensionality>&>(*arr.shape()));
            auto tmp_it = tmp.data();
            IterateBoostMultiArrayConcept(arr, [&power, &tmp_it](const typename T::element &item) {
                *tmp_it = std::pow(item, power);
                tmp_it++;
            });
            return tmp;
             */
        };

        template<class T>
        inline boost::multi_array<typename T::element, T::dimensionality> sqrt(const T &arr) {
            return SerialUnaryOperation(arr, [](const typename T::element &element) {
                return std::sqrt(element);
            });
            /*
            boost::multi_array<typename T::element, T::dimensionality> tmp;
            tmp.resize(reinterpret_cast<const boost::array<size_t, T::dimensionality>&>(*arr.shape()));
            auto tmp_it = tmp.data();
            IterateBoostMultiArrayConcept(arr, [&tmp_it](const typename T::element &item) {
                *tmp_it = static_cast<typename T::element>(std::sqrt(item));
                tmp_it++;
            });
            return tmp;
             */
        };

        template<class T>
        inline boost::multi_array<typename T::element, T::dimensionality> exp(const T &arr) {
            return SerialUnaryOperation(arr, [](const typename T::element &element) {
                return std::exp(element);
            });
        };

        template<class T>
        inline typename T::element sum(const T &arr) {
            typename T::element results = 0;
            IterateBoostMultiArrayConcept(arr, [&results](const typename T::element &item) {
                results += item;
            });
            return results;
        }

    }
}

#endif