//
// megrez_constants.hpp
//
// Created by Rixin Li on 4/18/17.
// Copyright © 2017 Rixin Li. All rights reserved.
//

#ifndef MEGREZ_MEGREZ_CONSTANTS_HPP
#define MEGREZ_MEGREZ_CONSTANTS_HPP

namespace megrez {

    namespace constants {

        /************************************/
        /********** math constants **********/
        /************************************/

        const double PI = 3.14159265358979323846;       // just pi

        /**************************/
        /********** mass **********/
        /**************************/

        const double M_sun = 1.9891e+30;                // in kg, solar mass
        const double M_jup = 1.8987e+27;                // in kg, Jupiter's mass
        const double M_earth = 5.9742e+24;              // in kg, earth mass
        const double m_p = 1.672621777e-27;             // in kg, proton's mass

        /****************************/
        /********** length **********/
        /****************************/

        const double au = 149597870700.0;               // in meters

        /**************************/
        /********** area **********/
        /**************************/

        const double sigma_H2 = 2e-19;                  // in m2, cross section of H2 molecular

        /**************************/
        /********** time **********/
        /**************************/

        const double year = 31557600.0;                 // in seconds


        /****************************************/
        /********** physical constants **********/
        /****************************************/

        const double G = 6.67384e-11;                   // in m3 / (kg s2), gravitational constant
        const double k_B = 1.3806488e-23;               // in J / K, Boltzmann constant

    }

}

#endif //MEGREZ_MEGREZ_CONSTANTS_HPP
