import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import scipy

from .constants import *


def compute_H(ssa, x):
    """Method for computing Chandrasekhar's H function, developed by Shuai Li

    Args:
        ssa (float or np.ndarray): single scattering albedo
        x (float): mu or mu0, i.e. cosines of incidence and emmittence angles
    """
    y = np.sqrt(1-ssa)
    r0 = (1-y)/(1+y)

    Hinv = 1 - (1 - y) * x * (r0 + (1 - 0.5 * r0 - r0 * x) * np.log((1+x)/x))
    return 1/Hinv


def compute_H2(ssa, x):
    """Alternative method for computing Chandrasekhar's H function, developed by Shuai Li

    Args:
        ssa (float or np.ndarray): single scattering albedo
        x (float): mu or mu0, i.e. cosines of incidence and emmittence angles
    """
    y = np.sqrt(1-ssa)

    arg = (1+x)/x
    a = x - 0.5 * x * np.log(arg) - x ** 2 * np.log(arg)
    b = x * np.log(arg)

    num = 1 + y
    den = 1 + y - a * (1 - y)**2 - b * (1 - y**2)

    return num/den


def compute_Bs(poros, g=np.pi/6, hs = None):
    """ Computes the opposition effect B(g), using Hapke 9.22

    Args:
        g (float or list_like, default = np.pi/6): phase angle

    Returns:
        B: opposition effect
    """
    if hs is None:
        O = poros
        B0 = 1
        h = -3/8 * np.log(1 - O)

        return B0/(1 + (1/h) * np.tan(g/2))
    else:
        return 1/(1 + np.tan(g/2)/hs)


def convert_oc_to_ssa(wl, n, k, D):
    """Converts empircal data about the optical constant to single scattering albedo as a function of wavelength

    Args:
        wl (nd.array): Wavelength in microns
        n (nd.array): Real index of refraction
        k (nd.array): Complex part of index of refraction
        D (np.float): particle size in microns

    Returns:
        _type_: _description_
    """

    alpha = 4 * np.pi * n * k/wl
    Dave = 2/3 * (n ** 2 - ((n ** 2 - 1) ** (3/2))/n) * D

    se = ((n - 1) ** 2 + k ** 2)/((n + 1) ** 2 + k ** 2) + 0.05

    si = 1 - 4/(n * (n + 1) ** 2)
    Theta = np.exp(-alpha * Dave)

    ssa = se + (1 - se) * ((1 - si) * Theta)/(1 - si * Theta)

    return ssa


def compute_mixed_P(Mis, ssas, Ps=[P_WAC, P_ICE], densities=[RHO_REG, RHO_ICE], diameters=[D_REG, D_ICE]):
    """ Compute the mixed phase function of different materials (especially ice and regolith). Default parameters
    contain values for phase function P, density, and diameter for regolith and ice, in that order.

    Args:
        Ps (list_like): Phase function P evaluated at angle g (standard is pi/6)
        Mis (list_like): Mass fractions, should sum to 1
        densities (list_like): Densities. Units are arbitrary as long as they are consistent between all densities.
        ssas (list_like): Single scattering albedo for each component. Each element in the list can be an nd_array containing ssa as a function of wavelength
        diameters (list_like): Average diameter for each component.

    Returns:
        Pout (float or list_like): Mixed Phase function value
    """
    num = 0
    den = 0

    for i in range(len(Ps)):
        P = Ps[i]
        M = Mis[i]
        rho = densities[i]
        ssa = ssas[i]
        d = diameters[i]

        num += M * P * ssa/(rho * d)
        den += M * ssa/(rho * d)

    return num/den


def compute_mixed_ssa(Mis, ssas, densities=[RHO_REG, RHO_ICE], diameters=[D_REG, D_ICE]):
    """ Compute the mixed single scattering albedo  of different materials (especially ice and regolith). Default parameters
    contain values density, and diameter for regolith and ice, in that order.

    Args:
        Mis (list_like): Mass fractions, should sum to 1
        densities (list_like): Densities. Units are arbitrary as long as they are consistent between all densities.
        ssas (list_like): Single scattering albedo for each component. Each element in the list can be an nd_array containing ssa as a function of wavelength
        diameters (list_like): Average diameter for each component.

    Returns:
        ssa_ave (float or list_like): Averaged single scattering albedo
    """
    num = 0
    den = 0

    for i in range(len(Mis)):
        M = Mis[i]
        rho = densities[i]
        ssa = ssas[i]
        d = diameters[i]

        num += M * ssa/(rho * d)
        den += M / (rho * d)
    return num/den


def compute_P(g, b, c, format='Legendre'):
    if format == 'Legendre':
        P = 1 + b * np.cos(g) + c * (1.5 * np.cos(g)**2 - 0.5)
    elif format == 'HenyeyGreensteinPascuzzo':
        P = (1-c) * (1 - b**2) / (1 - 2 * b * np.cos(g) + b**2) ** (3/2) + \
            c * (1 - b**2) / (1 + 2 * b * np.cos(g) + b**2)**(3/2)
    elif format == "HenyeyGreenstein":
        P = (1+c)/2 * (1 - b**2) / ((1 - 2 * b * np.cos(g) + b**2) ** (3/2)) + \
            (1-c)/2 * (1 - b**2) / ((1 + 2 * b * np.cos(g) + b**2)**(3/2))
        
    else:
        return None

    return P


def compute_P_material(g, material='WAC'):
    if material == 'regolith':
        # Mustard and Peters, 1981
        return compute_P(g, b=-0.4, c=0.25, format='Legendre')
    if material == 'ice':
        # Pascuzzo et al 2022
        return compute_P(g, b=0.5, c=0.8, format='HenyeyGreensteinPascuzzo')
    if material == 'WAC':
        b = 0.307
        c = 1
        return compute_P(g, b=b, c=c, format='HenyeyGreenstein')
    else:
        return None


def compute_Bc(g, hc=0.064/0.72):

    if isinstance(g, np.ndarray):
        out = []

        for gi in g:

            if gi == 0:
                out.append(1)
            else:

                zeta = np.tan(gi/2)/hc
                out.append((1 + (1 - np.exp(-zeta))/zeta)/(2 * (1 + zeta)**2))

        print(out)
        return np.array(out)
    else:
        if g == 0:
            return 1
        else:
            zeta = np.tan(g/2)/hc
            return ((1 + (1 - np.exp(-zeta))/zeta)/(2 * (1 + zeta)**2))
