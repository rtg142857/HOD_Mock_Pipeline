import numpy as np


def M0_function(magnitude, A, B):
    M0s = (A*(magnitude+20) + (B+11))
    M0s[M0s <=1.0] = 1.0
    return M0s

def sigma_function(magnitude, A, B, C, D):
    return A + (B-A) / (1.+np.exp((magnitude+C)*D))


def alpha_function(magnitude,A,B,C):
    return A + B ** (-magnitude - 20 + C)


def L_function(magnitude,A,B,C,D):
    return (A + 12) + B*(magnitude + 20) + C*(magnitude+20)**2 + D*(magnitude+20)**3



def spline_kernel_integral(x):
    """
    Returns the integral of the unscaled spline kernel function from -1 to x
    """
    if hasattr(x, "__len__"):
        # x in an array
        integral = np.zeros(len(x))
        absx = abs(x)
        ind = absx < 0.5
        integral[ind] = absx[ind] - 2*absx[ind]**3 + 1.5*absx[ind]**4
        ind = np.logical_and(absx >= 0.5, absx < 1.)
        integral[ind] = 0.375 - 0.5*(1-absx[ind])**4
        ind = absx >= 1.
        integral[ind] = 0.375
        ind = x < 0
        integral[ind] = -integral[ind]
    else:
        # x is a number
        absx = abs(x)
        if   absx < 0.5: integral = absx - 2*absx**3 + 1.5*absx**4
        elif absx < 1:   integral = 0.375 - 0.5*(1-absx)**4
        else:            integral = 0.375
        if x < 0: integral = -integral
    return integral

def cumulative_spline_kernel(x, mean=0, sig=1):
    """
    Returns the integral of the rescaled spline kernel function from -inf to x.
    The spline kernel is rescaled to have the specified mean and standard
    deviation, and is normalized.
    """
    integral = spline_kernel_integral((x-mean)/(sig*np.sqrt(12))) / 0.75
    y = 0.5 * (1. + 2*integral)
    return y


def Cen_HOD(params,mass_bins):
    """
    Returns the HOD for central galaxies at the mass bins provided
    """
    Mmin, sigma_logm = params[:2]
    result = cumulative_spline_kernel(np.log10(mass_bins), mean = Mmin, sig=sigma_logm/np.sqrt(2))
    return(result)

def Sat_HOD(params,cen_hod,mass_bins):
    """
    Returns the HOD for satellite galaxies
    """
    M0, M1, alpha = params[2:].copy()
    M0 = 10**M0
    M1 = 10**M1
    result = cen_hod * (((mass_bins-M0)/M1)**alpha)
    return(result)



