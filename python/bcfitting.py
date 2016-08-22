# -*- coding: utf-8 -*-
"""
BlackChirp Python Module
Author: Kyle N. Crabtree
"""

import numpy
import scipy.optimize as spopt

def bc_gauss(x, A, x0, w):
    """
    Calculate value of amplitude-normalized Gaussian peak at x:
    
        f(x) = A * exp(-(x-x0)^2/2w^2)
    
    Arguments:
    
    x : float
        Dependent variable for Gaussian
    A : float
        Amplitude of Gaussian
    x0 : float
        Center position of Gaussian (i.e., f(x0) = A)
    w : float
        Standard deviation of Gaussian (FWHM = 2sqrt(2ln2)*w)
        
    Returns:
    
    f : float
        Value of Gaussian function at position x
    """
    
    return A*numpy.exp(-((x-x0)/w)**2./2.)
    
def bc_multigauss(x, *args):
    """
    Calculate an array consisting of a sum of Gaussians with baseline
    
    This function is used in peak fitting to handle an arbitrary number
    of Gaussian peaks. The fit parameters are passed in as a packed tuple.
    The tuple must contain 3n + 1 parameters: element 0 is the baseline
    offset term, and then for each peak the tuple contains values for
    A, x0, and w.
    
    Calculation: y0 + \sum_i^N bc_gauss(Ai,x0i,wi), where N is the number
    of peaks
    
    Arguments:
    
    x : array-like
        The x-values at which to calculate the function
        
    args : tuple
        A tuple that consists of (y0, A1, x01, w1, A2, x02, w2, ...)
        
    Returns:
    
    fdata : 1D array
        Array containing values of multiple gaussian function at x points
        
    """
    
    out = numpy.empty(len(x))
    
    for i in range(len(x)):
        out[i] = args[0]
        for j in range(0, (len(args)-1)//3):
            out[i] += bc_gauss(x[i], args[3*j+1], args[3*j+2], args[3*j+3])
        
    return out
    
def bc_multigauss_jac(*params):
    """
    Calculate Jacobian matrix for multi gaussian function
    
    NOTE: Though faster, the results seem to be worse than when finite
    differences are used.
    
    Arguments:
    
    params : tuple
        Tuple containing:
            0: fit parameters (see bc_multigauss)
            1: x data (array-like)
            2: y data (array-like)
            3: function (bc_multigauss)
        
    Returns:
    
    jac : 2D array
        2D array (len(x) by len(params)) where element (row, col) =(i,j)
        contains df(x_i)/dparam_j
    """
    
    p = params[0]
    x = params[1]
    out = numpy.zeros((len(x),len(p)))
    for i in range(len(x)):
        out[i][0] = 1.
        for j in range((len(p)-1)//3):
            gval = bc_gauss(x[i],p[3*j+1],p[3*j+2],p[3*j+3])
            out[i][3*j+1] = gval/p[3*j+1]
            out[i][3*j+2] = gval*(x[i]-p[3*j+2])/p[3*j+3]**2.
            out[i][3*j+3] = gval*(x[i]-p[3*j+2])**2./p[3*j+3]**3.
    
    return out  

def bc_gauss_area(x, A, x0, w):
    """
    Calculate value of area-normalized Gaussian peak at x:
    
        f(x) = A/(w*sqrt(2pi)) * exp(-(x-x0)^2/2w^2)
    
    Arguments:
    
    x : float
        Dependent variable for Gaussian
    A : float
        Area of Gaussian
    x0 : float
        Center position of Gaussian (i.e., f(x0) = A)
    w : float
        Standard deviation of Gaussian (FWHM = 2sqrt(2ln2)*w)
        
    Returns:
    
    f : float
        Value of Gaussian function at position x
    """
    
    return A/w/numpy.sqrt(2*numpy.pi)*numpy.exp(-((x-x0)/w)**2./2.)
    
def bc_multigauss_area(x, *args):
    """
    Calculate an array consisting of a sum of Gaussians with baseline
    
    This function is used in peak fitting to handle an arbitrary number
    of Gaussian peaks. The fit parameters are passed in as a packed tuple.
    The tuple must contain 3n + 1 parameters: element 0 is the baseline
    offset term, and then for each peak the tuple contains values for
    A, x0, and w.
    
    Calculation: y0 + \sum_i^N bc_gauss(Ai,x0i,wi), where N is the number
    of peaks
    
    Arguments:
    
    x : array-like
        The x-values at which to calculate the function
        
    args : tuple
        A tuple that consists of (y0, A1, x01, w1, A2, x02, w2, ...)
        
    Returns:
    
    fdata : 1D array
        Array containing values of multiple gaussian function at x points
        
    """
    
    out = numpy.empty(len(x))
    
    for i in range(len(x)):
        out[i] = args[0]
        for j in range(0, (len(args)-1)//3):
            out[i] += bc_gauss_area(x[i], args[3*j+1], args[3*j+2],
                                     args[3*j+3])
        
    return out
    
def bc_multigauss_area_jac(*params):
    """
    Calculate Jacobian matrix for multi gaussian function
    
    NOTE: Though faster, the results seem to be worse than when finite
    differences are used.
    
    Arguments:
    
    params : tuple
        Tuple containing:
            0: fit parameters (see bc_multigauss)
            1: x data (array-like)
            2: y data (array-like)
            3: function (bc_multigauss)
        
    Returns:
    
    jac : 2D array
        2D array (len(x) by len(params)) where element (row, col) =(i,j)
        contains df(x_i)/dparam_j
    """
    
    p = params[0]
    x = params[1]
    out = numpy.zeros((len(x),len(p)))
    for i in range(len(x)):
        out[i][0] = 1.
        for j in range((len(p)-1)//3):
            gval = bc_gauss_area(x[i],p[3*j+1],p[3*j+2],p[3*j+3])
            out[i][3*j+1] = gval/p[3*j+1]
            out[i][3*j+2] = gval*(x[i]-p[3*j+2])/p[3*j+3]**2.
            out[i][3*j+3] = gval*((x[i]-p[3*j+2])**2./p[3*j+3]**3.
                                  - p[3*j+3]**-1.)
    
    return out        
    
    
def bc_fit_function(params, x, y, function):
    """
    Calculate deviation of function from y data.
    
    This formulation of the fit function is required for the least squares
    minimization
    
    Arguments:
    
    params : tuple
        Fit parameters for function.
        
    x : array-like
        X values for fiting
        
    y : array-like
        Y values for fitting
        
    function : callable
        A callable function that takes (x,params) as arguments and returns
        an array containing the values of the function at the x positions
        
    """
    
    return function(x,*params) - y
    
    
def fit_peaks_gauss(*args):
    """
    Fit yarray = func(xarray, params)
    
    This function does a least squares fit of the yarray to a sum of
    gaussians (number determined by params). Internally, this function uses
    scipy.optimize.leastsq to compute optimize the fit parameters and
    calculate uncertainties.
    
    Arguments:
    
    *args : tuple (xarray, yarray, params, bounds)
    
    xarray : array-like
        X values for fitting
        
    yarray : array-like
        Data to fit
        
    params : tuple
        Parameters to fit (must be appropriate for func)
        
    bounds : Array of tuples
        Constraints on parameters. The first tuple contains minimum
        bounds, the second contains maximum bounds
        
    Returns:
        
    """
    
    xarray = args[0]
    yarray = args[1]
    params = args[2]
    bounds = args[3]
    

    
    if bounds is not None:
        res = spopt.curve_fit(bc_multigauss,xarray,yarray,p0=params,
                              bounds=bounds)
    else:
        res = spopt.curve_fit(bc_multigauss,xarray,yarray,
                                  p0=params,full_output=True)

    perr = numpy.sqrt(numpy.diag(res[1]))
    
    return res[0], res[1], perr
    
    
def fit_peaks_gauss_area(*args):
    """
    Fit yarray = func(xarray, params)
    
    This function does a least squares fit of the yarray to a sum of
    gaussians (number determined by params). Internally, this function uses
    scipy.optimize.leastsq to compute optimize the fit parameters and
    calculate uncertainties.
    
    Arguments:
    
    *args : tuple (xarray, yarray, params, bounds)
    
    xarray : array-like
        X values for fitting
        
    yarray : array-like
        Data to fit
        
    params : tuple
        Parameters to fit (must be appropriate for func)
        
    bounds : Array of tuples
        Constraints on parameters. The first tuple contains minimum
        bounds, the second contains maximum bounds
        
    **kwargs : dict. Recognized keys:
    
        "method" : "area" or "amplitude"
        
    Returns:
        
    """
    
    xarray = args[0]
    yarray = args[1]
    params = args[2]
    bounds = args[3]

    
    if bounds is not None:
        res = spopt.curve_fit(bc_multigauss_area,xarray,yarray,p0=params,
                              bounds=bounds,jac=bc_multigauss_jac)
    else:
        res = spopt.curve_fit(bc_multigauss_area,xarray,yarray,
                                  p0=params,full_output=True)

    perr = numpy.sqrt(numpy.diag(res[1]))
    
    return res[0], res[1], perr