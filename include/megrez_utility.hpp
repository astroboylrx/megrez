//
// megrez_utility.hpp
//
// Created by Rixin Li on 4/18/2017.
// Copyright © 2017 Rixin Li. All rights reserved.
//

#ifndef MEGREZ_MEGREZ_UTILITY_HPP
#define MEGREZ_MEGREZ_UTILITY_HPP

// Include header files from the same project

// Include header files from non-standard projects
#include "boost/multi_array.hpp"
// Include C/C++ libraries
#include <algorithm>
#include <limits>                       // std::numeric_limits
#include <type_traits>                  // std::is_integer, std::is_signed
// Include other libraries


namespace megrez {

    /*! \struct template <bool, class T, class U> struct __SelectIf_base
     *  \brief a template for struct __SelectIF to accept a bool as condition */
    template <bool, class T, class U>
    struct __SelectIf {};

    /*! \struct template <class T, class U> struct __SelectIf_true
     *  \brief a template for struct __SelectIF to select T type if bool is true */
    template <class T, class U>
    struct __SelectIf<true, T, U> { typedef T type; };

    /*! \struct template <class T, class U> struct __SelectIf_false
     *  \brief a template for struct __SelectIF to select U type if bool is false */
    template <class T, class U>
    struct __SelectIf<false, T, U> { typedef U type; };

    /*! \struct template <class T, class U> struct PromoteNumeric
     *  \brief this struct controls the type promotion, this struct nests many levels of selections. Read comments/explanations from inside and notice that comments start with new lines are different with comments after statements */
    template <class T, class U>
    struct PromoteNumeric {
        typedef typename __SelectIf<
                //if T and U are both integers or both non-integers
                std::numeric_limits<T>::is_integer == std::numeric_limits<U>::is_integer, // middle, served as bool for outermost
                //then pick the larger type
                typename __SelectIf<(sizeof(T) > sizeof(U)), T,
                        //else if they are equal
                        typename __SelectIf<(sizeof(T) == sizeof(U)),
                                //pick the one which is unsigned
                                typename __SelectIf<std::numeric_limits<T>::is_signed, U, T>::type, // this is the innermost layer
                                //else select U as bigger
                                U
                        >::type // this is the second innermost layer
                >::type, // middle, served as T for outermost, nests other layers
                //else pick the one which is not integer
                typename __SelectIf<std::numeric_limits<T>::is_integer, U, T>::type // middle, served as U for outermost
        >::type type; // outermost layer
    };


    /*! \fn template <typename T> T MaxOf(const T &a, const T &b)
 *  \brief return the larger one of a and b */
    template <typename T>
    T MaxOf(const T &a, const T &b) {
        return std::max(a, b);
    }

/*! \fn template <typename T, typename... Args> T MaxOf(const T &a, const T &b, Args... args)
 *  \brief return the maximum one in a list, downgrade to MaxOf(a, b) */
    template <typename T, typename... Args>
    T MaxOf(const T &a, const T &b, Args... args) {
        return MaxOf(std::max(a, b), args...);
    }

/*! \fn template <typename T> T MinOf(const T &a, const T &b)
 *  \brief return the smaller one of a and b */
    template <typename T>
    T MinOf(const T &a, const T &b) {
        return std::min(a, b);
    }

/*! \fn template <typename T, typename... Args> T MinOf(const T &a, const T &b, Args... args)
 *  \brief return the minimum one in a list, downgrade to MaxOf(a, b) */
    template <typename T, typename... Args>
    T MinOf(const T &a, const T &b, Args... args) {
        return MaxOf(std::min(a, b), args...);
    }


/*! \namespace sn
 *  \brief store simplified names */
    namespace sn {

        /*! \alias using bma_range = boost::multi_array_types::index_range
         *  \brief used to generate boost index range */
        using bma_range = boost::multi_array_types::index_range;

        using darr1d = boost::multi_array<double, 1>;
        using darr2d = boost::multi_array<double, 2>;
        using darr3d = boost::multi_array<double, 3>;

        using dshape1d = boost::array<darr1d::index, 1>;
        using dshape2d = boost::array<darr2d::index, 2>;
        using dshape3d = boost::array<darr3d::index, 3>;

        using dview1d = darr1d::array_view<1>::type;
        using dview2d = darr2d::array_view<2>::type;
        using dslice2d = darr2d::array_view<1>::type;
        using dview3d = darr3d::array_view<3>::type;
        using dslice3d = darr3d::array_view<2>::type;
        using dpencil3d = darr3d::array_view<1>::type;
    }

}


#endif //MEGREZ_MEGREZ_UTILITY_HPP
