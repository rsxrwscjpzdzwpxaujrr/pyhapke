import numpy as np
import scipy.optimize
import scipy
from .constants import *
from .hapkeFuncs import *


class HapkeRTM:
    def __init__(
        self,
        i=0,
        e=np.pi / 6,
        g=np.pi / 6,
        P=P_WAC,
        wl=None,
        poros=0.41,
        Bs0=1,
        Bc0=0,
        hs=None,
    ):
        """Initialize an instance of the Hapke Radiative Transfer Model, which has tools to calculate ssa from reflectance, and vice versa

        Args:
            i (float, optional): Incidence angle in radians. Defaults to np.pi/6.
            e (float, optional): Emmitance angle in radians. Defaults to 0.
            g (float, optional): Phase angle in radiance. Defaults to np.pi/6.
            P (float or nd.array, optional): Phase function value at angle g.  Defaults to P_REG, regolith at 30 deg phase. Can also be input
            as an array with respect to wavelength
            wl (nd.array, optional): Wavelengths corresponding to each entry in P, if P is a nd.array. Defaults to None.
            poros (float, optional): porosity if using compute_B to calculate SHOE backscatter
            Bs0 (float, optional): S type backscatter effect coefficient
            Bc0 (float, optional): coherent backscatter effect coefficient
        """
        self.mu0 = np.cos(i)
        self.mu = np.cos(e)
        self.hs = hs
        self.Bs = compute_Bs(
            poros=poros, g=g, hs=hs
        )  # "Classic" SHOE version of backscatter
        self.Bs0 = Bs0
        self.Bc0 = Bc0
        self.Bc = compute_Bc(g)
        # print(self.B)
        # self.B = 0.4248
        self.P = P
        self.wl = wl
        self.hapke_polynomial = None
        self.g = g

        pass

    def hapke_function_REFF(self, ssa):
        """Function R(omega), assuming other parameters are known. This is
        the REFF version of the function (Hapke equation 10.4), computing the reflectance factor (reflectance coefficient)

        Args:
            ssa (float or nd.array): single scattering albedo

        Returns:
            float or nd.array: Reflectance factor
        """

        mu0 = self.mu0
        mu = self.mu
        P = self.P
        Bs0 = self.Bs0
        Bc0 = self.Bc0
        Bc = self.Bc
        Bs = self.Bs

        # Compute H-funct
        H = compute_H2(ssa, mu)
        H0 = compute_H2(ssa, mu0)

        # R = (ssa/4) * mu0 / (mu0 + mu) * ((1 + B) * P + H * H0 - 1)

        R = (
            (ssa / 4)
            / (mu0 + mu)
            * (((1 + Bs * Bs0) * P) + (H * H0 - 1))
            * (1 + Bc * Bc0)
        )

        return R

    def hapke_function_RADF(self, ssa):
        """Function R(omega), assuming other parameters are known. This is
        the RADF version of the function (Hapke equation 10.5), which computes I/F radiance.

        Args:
            ssa (float or nd.array): single scattering albedo
            OE: Type of backscatter opposition effect. Use None for lab data, CBOE for WAC albedo

        Returns:
            float or nd.array: I/F radiance
        """
        mu0 = self.mu0
        mu = self.mu
        P = self.P
        Bs0 = self.Bs0
        Bc0 = self.Bc0
        Bc = self.Bc
        Bs = self.Bs

        # Compute H-funct
        H = compute_H(ssa, mu)
        H0 = compute_H(ssa, mu0)

        # R = (ssa/4) * mu0 / (mu0 + mu) * ((1 + B) * P + H * H0 - 1)

        R = (
            (ssa / 4)
            * mu0
            / (mu0 + mu)
            * ((1 + Bs * Bs0) * P + H * H0 - 1)
            * (1 + Bc * Bc0)
        )

        return R

    def hapke_function_RedR(self, ssa):
        """Function R_r(omega), assuming other parameters are known. This is
        equal to RADF/ the Lommel Seeliger factor

        Args:
            ssa (float or nd.array): single scattering albedo

        Returns:
            float or nd.array: I/F radiance
        """
        mu0 = self.mu0
        mu = self.mu
        P = self.P
        Bs0 = self.Bs0
        Bc0 = self.Bc0
        Bc = self.Bc
        Bs = self.Bs

        """if OE == "CBOE":
            Bs0 = 0
            Bc0 = 0.75
        elif OE == "SHOE":
            Bs0 = 0.293
            Bc0 = .456"""

        # Compute H-funct
        H = compute_H(ssa, mu)
        H0 = compute_H(ssa, mu0)

        # R = (ssa/4) * mu0 / (mu0 + mu) * ((1 + B) * P + H * H0 - 1)
        R = (ssa / 4) * ((1 + Bs * Bs0) * P + H * H0 - 1) * (1 + Bc * Bc0)

        return R

    def hapke_function_BRDF(self, ssa):
        """Function R(omega), assuming other parameters are known. This is
        the BRDF version of the function (Hapke equation 10.5), computing Bi-Directional Reflectance

        Args:
            ssa (float or nd.array): single scattering albedo

        Returns:
            float or nd.array: Bidirectional reflectance distribution factor
        """
        mu0 = self.mu0
        mu = self.mu
        P = self.P
        Bs0 = self.Bs0
        Bc0 = self.Bc0
        Bc = self.Bc
        Bs = self.Bs

        # Compute H-funct
        H = compute_H2(ssa, mu)
        H0 = compute_H2(ssa, mu0)

        # R = (ssa/4) * mu0 / (mu0 + mu) * ((1 + B) * P + H * H0 - 1)

        R = (ssa / 4) / (mu0 + mu) * ((1 + Bs * Bs0) * P + H * H0 - 1) * (1 + Bc * Bc0)

        return R

    def hapke_function(self, ssa):
        """Function R(omega), assuming other parameters are known, where R is bidirectional reflectance

        Args:
            ssa (float or nd.array): single scattering albedo

        Returns:
            float or nd.array: Bidirectional reflectance
        """
        mu0 = self.mu0
        mu = self.mu
        P = self.P
        Bs0 = self.Bs0
        Bc0 = self.Bc0
        Bc = self.Bc
        Bs = self.Bs
        # Compute H-funct
        H = compute_H2(ssa, mu)
        H0 = compute_H2(ssa, mu0)

        # R = (ssa/4) * mu0 / (mu0 + mu) * ((1 + B) * P + H * H0 - 1)
        R = (
            (ssa / 4 / np.pi)
            * mu0
            / (mu0 + mu)
            * ((1 + Bs * Bs0) * P + H * H0 - 1)
            * (1 + Bc * Bc0)
        )

        return R

    def compute_ssa_from_R(self, R, method="lm", model="RADF"):
        """Compute single scattering albedo given reflectance or radiance and input. By default, uses Levenberg-Marquardt algorithm as part of
        scipy.optimize.root. Hybr method (scipy default) not recommended, due to difficulty of finding simualted jacobian.

        Args:
            R (float or nd.array): Reflectance
            method (str, optional): Root finding method from scipy.optimize.root. Defaults to 'lm'.
            model (function, optional): Version of Hapke's model to compute ssa from. Defaults to hapke_function_REFF.
        """

        if model == "REFF":

            def obj_func(ssa, R):
                Rpred = self.hapke_function_REFF(ssa)
                return Rpred - R

        elif model == "BRDF":

            def obj_func(ssa, R):
                Rpred = self.hapke_function_BRDF(ssa)
                return Rpred - R

        elif model == "RADF":

            def obj_func(ssa, R):
                Rpred = self.hapke_function_RADF(ssa)
                return Rpred - R

        elif model == "RedR":

            def obj_func(ssa, R):
                Rpred = self.hapke_function_RedR(ssa)
                return Rpred - R

        elif model is None:

            def obj_func(ssa, R):
                Rpred = self.hapke_function(ssa)
                return Rpred - R

        if type(R) is np.ndarray:
            x0 = np.ones_like(R) * 0.5
        else:
            x0 = 0.5

        sol = scipy.optimize.root(
            fun=obj_func, x0=x0, args=(R), method=method, tol=1e-15
        )

        return sol.x

    # def calculate_hapke_polynomial(self, )
