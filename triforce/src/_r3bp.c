#include "dopri/dop853.h"
#include "math.h"

void r3bp_derivs(unsigned ndim, double t, double *y, double *dydx,
                 GradFn func, double *pars, unsigned norbits) {
    double q = pars[0];
    double ecc = pars[1];
    double nu = pars[2];

    double u2 = q/(1.+q);
    double u1 = 1.0-u2;
    double a  = 1.0;
    double n  = 1./sqrt(a*a*a);

    double r1 = sqrt((y[0] + u2)*(y[0] + u2) + y[1]*y[1]);
    double r2 = sqrt((y[0] - u1)*(y[0] - u1) + y[1]*y[1]);
    double dUdx1 = n*n*y[0] - (u2*(y[0] - u1))/(r2*r2*r2) - (u1*(u2 + y[0]))/(r1*r1*r1);
    double dUdx2 = n*n*y[1] - (u2*y[1])/(r2*r2*r2) - (u1*y[1])/(r1*r1*r1);

    //Viscosity
    double rr  = sqrt(y[0]*y[0] + y[1]*y[1]);
    double OmK = pow(rr,-1.5);
    double rho = 1.0;
    //double nu = 0.00025;
    //double Fvisc_phi = -9./4.*rho*nu*(OmK - n)/(rr);

    double FK = -9./4.*rho*nu;
    //double Fvisc_xi = -Fvisc_phi/rr * (y[0]*sin(x) + y[1]*cos(x));
    //double Fvisc_et = Fvisc_phi/rr *  (y[0]*cos(x) - y[1]*sin(x));

    double FvscRx = FK/(rr*rr)*(y[3] - n*y[1]);
    double FvscRy = FK/(rr*rr)*(y[4] + n*y[0]);

    //double FvscRx = -Fvisc_phi*y[1]/rr;  //Fvisc_xi *cos(x) + Fvisc_et *sin(x);
    //double FvscRy = Fvisc_phi*y[0]/rr;   //Fvisc_et *cos(x) - Fvisc_xi *sin(x);

    double rsep = 1./(1. + ecc*cos(t));

    double a1 = dUdx1*rsep + 2.*n*y[4] + FvscRx;
    double a2 = dUdx2*rsep - 2.*n*y[3] + FvscRy;
    double a3 = 0.0; ///work in the binary plane

    // SET DERIVATIVES
    dydx[0] = y[3];        // dx/dt = vx
    dydx[1] = y[4];        // dy/dt = vy
    dydx[2] = 0.;        // dz/dt = vz, in the binary plane
    dydx[3] = a1;         // dvx/dt = ax
    dydx[4] = a2;         // dvy/dt = ay
    dydx[5] = 0.;         // dvz/dt = az, in the binary plane
}

// Straight-up kepler
// void r3bp_derivs(unsigned ndim, double t, double *y, double *dydx,
//                  GradFn func, double *pars, unsigned norbits) {
//     double q = pars[0];
//     double ecc = pars[1];
//     double nu = pars[2];

//     double ax = y[0] * pow(y[0]*y[0] + y[1]*y[1], -1.5);
//     double ay = y[1] * pow(y[0]*y[0] + y[1]*y[1], -1.5);

//     // SET DERIVATIVES
//     dydx[0] = y[3];        // dx/dt = vx
//     dydx[1] = y[4];        // dy/dt = vy
//     dydx[2] = 0.;        // dz/dt = vz
//     dydx[3] = -ax;         // dvx/dt = ax
//     dydx[4] = -ay;         // dvy/dt = ay
//     dydx[5] = 0.;         // dvz/dt = az
// }

// rotating kepler
// void r3bp_derivs(unsigned ndim, double t, double *y, double *dydx,
//                  GradFn func, double *pars, unsigned norbits) {
//     double q = pars[0];
//     double ecc = pars[1];
//     double nu = pars[2];

//     double ax = y[0] * pow(y[0]*y[0] + y[1]*y[1], -1.5) + y[0] - 2.*y[4];
//     double ay = y[1] * pow(y[0]*y[0] + y[1]*y[1], -1.5) + y[1] + 2.*y[3];

//     // SET DERIVATIVES
//     dydx[0] = y[3];        // dx/dt = vx
//     dydx[1] = y[4];        // dy/dt = vy
//     dydx[2] = 0.;        // dz/dt = vz
//     dydx[3] = -ax;         // dvx/dt = ax
//     dydx[4] = -ay;         // dvy/dt = ay
//     dydx[5] = 0.;         // dvz/dt = az
// }

// APW attempt at CR3BP
// void r3bp_derivs(unsigned ndim, double t, double *w, double *wdot,
//                  GradFn func, double *pars, unsigned norbits) {
//     double q = pars[0];
//     double ecc = pars[1];
//     double nu = pars[2];

//     double r1 = q / (1+q);
//     double r2 = 1 / (1+q);
//     double GM1 = r2;
//     double GM2 = r1;
//     double Om = 1.;

//     double x = w[0];
//     double y = w[1];
//     // double z = w[2];
//     double vx = w[3];
//     double vy = w[4];
//     // double vz = w[5];

//     double u1 = pow((x+r1)*(x+r1) + y*y, 1.5);
//     double u2 = pow((x-r2)*(x-r2) + y*y, 1.5);

//     double ax = -GM1*(x+r1) / u1 - GM2*(x-r2) / u2 + 0.5*Om*Om*x + 2*Om*vy;
//     double ay = -GM1*y/u1 - GM2*y/u2 + 0.5*Om*Om*y - 2*Om*vx;

//     // SET DERIVATIVES
//     wdot[0] = vx;
//     wdot[1] = vy;
//     wdot[2] = 0.;
//     wdot[3] = ax;
//     wdot[4] = ay;
//     wdot[5] = 0.;
// }
