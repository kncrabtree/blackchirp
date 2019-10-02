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
    
    out = numpy.zeros(len(x))
    out += args[0]
    
    for j in range(0, (len(args)-1)//3):
        out += bc_gauss(x, args[3*j+1], args[3*j+2], args[3*j+3])
        
    return out


def bc_multigauss_fixw(x, *args):
    """
    Calculate an array consisting of a sum of Gaussians with baseline
    
    This function is used in peak fitting to handle an arbitrary number
    of Gaussian peaks with a common linewidth.
    
    The fit parameters are passed in as a packed tuple.
    The tuple must contain 2n + 2 parameters: element 0 is the baseline
    offset term, 1 is the width, and then for each peak the tuple contains
    values for A and x0.
    
    Calculation: y0 + \sum_i^N bc_gauss(Ai,x0i,w), where N is the number
    of peaks
    
    Arguments:
    
    x : array-like
        The x-values at which to calculate the function
        
    args : tuple
        A tuple that consists of (y0, w, A1, x01, A2, x02, ...)
        
    Returns:
    
    fdata : 1D array
        Array containing values of multiple gaussian function at x points
        
    """
    
    out = numpy.zeros(len(x))
    out += args[0]

    for j in range(0, (len(args)-2)//2):
        out += bc_gauss(x, args[2*j+2], args[2*j+3], args[1])
        
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
    
    out = numpy.zeros(len(x))
    out += args[0]
    for j in range(0, (len(args)-1)//3):
        out += bc_gauss_area(x, args[3*j+1], args[3*j+2],
                                     args[3*j+3])
        
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
    
    
def fit_peaks(*args):
    """
    Fit yarray = func(xarray, params)
    
    This function does a least squares fit of the yarray to a sum of
    gaussians (number determined by params). Internally, this function uses
    scipy.optimize.leastsq to compute optimize the fit parameters and
    calculate uncertainties.
    
    Arguments:
    
    *args : tuple (func, xarray, yarray, params, bounds)
    
    func : function
        The function to use for fitting
    
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
    
    func = args[0]
    xarray = args[1]
    yarray = args[2]
    params = args[3]
    bounds = args[4]
    

    
    if bounds is not None:
        res = spopt.curve_fit(func,xarray,yarray,p0=params,
                              bounds=bounds)
    else:
        res = spopt.curve_fit(func,xarray,yarray,
                                  p0=params,full_output=True)

    perr = numpy.sqrt(numpy.diag(res[1]))
    
    return res[0], res[1], perr
