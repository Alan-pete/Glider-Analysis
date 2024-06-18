from scipy.linalg import eig
import numpy as np
from matplotlib import pyplot as plt
import json
import math

class Dynamic_mode_analysis:

    def __init__(self, filename):

        json_data = json.loads(open(filename).read())  # reads in json file for values defined below

        # Constant values (given or assumed) for calculations
        self.g = 32.174  # gravitational acceleration (ft/s)
        self.theta0 = 0.0  # level flight at launch
        self.xt0 = 0.0  # no thrust
        self.zt0 = 0.0
        self.alpha_t0 = 0.0

        # Aircraft inputs
        self.Sw = json_data["aircraft"]["wing_area[ft^2]"]  # area of main wing
        self.bw = json_data["aircraft"]["wing_span[ft]"]  # span of main wing
        self.cw = self.Sw/self.bw  # average chord of main wing
        self.w = json_data["aircraft"]["weight[lbf]"]  # weight of the aircraft

        self.Ixxb = json_data["aircraft"]["Ixx[slug-ft^2]"]  # inertia values
        self.Iyyb = json_data["aircraft"]["Iyy[slug-ft^2]"]
        self.Izzb = json_data["aircraft"]["Izz[slug-ft^2]"]
        self.Ixzb = json_data["aircraft"]["Ixz[slug-ft^2]"]
        self.Ixyb = json_data["aircraft"]["Ixy[slug-ft^2]"]
        self.Iyzb = json_data["aircraft"]["Iyz[slug-ft^2]"]
        self.hx = json_data["aircraft"]["hx[slug-ft^2/s]"]  # angular momentum values
        self.hy = json_data["aircraft"]["hy[slug-ft^2/s]"]
        self.hz = json_data["aircraft"]["hz[slug-ft^2/s]"]

        # Initial values
        self.airspeed = json_data["initial"]["airspeed[ft/s]"]
        self.altitude = json_data["initial"]["altitude[ft]"]
        self.heading = json_data["initial"]["heading[deg]"]
        self.elevation_angle = json_data["initial"]["state"]["elevation_angle[deg]"]
        self.bank_angle = json_data["initial"]["state"]["bank_angle[deg]"]
        self.alpha = json_data["initial"]["state"]["alpha[deg]"]
        self.beta = json_data["initial"]["state"]["beta[deg]"]
        self.p = json_data["initial"]["state"]["p[deg/s]"]
        self.q = json_data["initial"]["state"]["q[deg/s]"]
        self.r = json_data["initial"]["state"]["r[deg/s]"]

        # Analysis values
        self.rho = json_data["analysis"]["density[slugs/ft^3]"]

        # Aerodynamic values
        self.ground_effect = json_data["aerodynamics"]["ground_effect"]["use_ground_effect"]  # ground effect
        self.r_t = json_data["aerodynamics"]["ground_effect"]["taper_ratio"]
        self.gust = json_data["aerodynamics"]["gust_magnitude[ft/s]"]
        self.stall = json_data["aerodynamics"]["stall"]["use_stall_model"]
        self.alpha_blend = json_data["aerodynamics"]["stall"]["alpha_blend[deg]"]
        self.blend_factor = json_data["aerodynamics"]["stall"]["blending_factor"]

        # CL
        CL = json_data["aerodynamics"]["CL"]
        self.CL_0 = CL["0"]
        self.CL_alpha = CL["alpha"]
        self.CL_alpha_hat = CL["alpha_hat"]
        self.CL_qbar = CL["qbar"]
        self.CL_de = CL["de"]

        # CS
        CS = json_data["aerodynamics"]["CS"]
        self.CS_beta = CS["beta"]
        self.CS_pbar = CS["pbar"]
        self.CS_Lpbar = CS["Lpbar"]
        self.CS_rbar = CS["rbar"]
        self.CS_da = CS["da"]
        self.CS_dr = CS["dr"]

        # CD
        CD = json_data["aerodynamics"]["CD"]
        self.CD_L0 = CD["L0"]
        self.CD_L = CD["L"]
        self.CD_L2 = CD["L2"]
        self.CD_S2 = CD["S2"]
        self.CD_Lqbar = CD["Lqbar"]
        self.CD_L2qbar = CD["L2qbar"]
        self.CD_de = CD["de"]
        self.CD_Lde = CD["Lde"]
        self.CD_de2 = CD["de2"]

        # Cl (don't confuse w/ CL)
        Cl = json_data["aerodynamics"]["Cl"]
        self.Cl_beta = Cl["beta"]
        self.Cl_pbar = Cl["pbar"]
        self.Cl_Lrbar = Cl["Lrbar"]
        self.Cl_da = Cl["da"]
        self.Cl_dr = Cl["dr"]

        # Cm
        Cm = json_data["aerodynamics"]["Cm"]
        self.Cm_0 = Cm["0"]
        self.Cm_alpha = Cm["alpha"]
        self.Cm_alpha_hat = Cm["alpha_hat"]
        self.Cm_qbar = Cm["qbar"]
        self.Cm_de = Cm["de"]

        # Cn
        Cn = json_data["aerodynamics"]["Cn"]
        self.Cn_beta = Cn["beta"]
        self.Cn_Lpbar = Cn["Lpbar"]
        self.Cn_rbar = Cn["rbar"]
        self.Cn_da = Cn["da"]
        self.Cn_Lda = Cn["Lda"]
        self.Cn_dr = Cn["dr"]

        # fixing some derivative values
        self.CD_qbar = (self.CD_L2qbar * self.CL_0**2) + (self.CD_Lqbar * self.CL_0) + CD["qbar"]
        self.Cl_rbar = (self.Cl_Lrbar * self.CL_0) + Cl["rbar"]
        self.Cn_pbar = (self.Cn_Lpbar * self.CL_0) + Cn["pbar"]
        self.CY_pbar = (self.CS_Lpbar * self.CL_0) + CS["pbar"]

        # calculating unknown values for matrices
        self.CD_o = self.CD_L0 + (self.CD_L * self.CL_0) + (self.CD_L2 * self.CL_0**2)
        self.CD_alpha = (self.CD_L * self.CL_alpha) + (2 * self.CD_L2 * self.CL_0 * self.CL_alpha)
        self.V_0 = np.sqrt(self.w / (0.5 * self.CL_0 * self.rho * self.Sw))  # equilibrium velocity
        self.R_rho_x = (4 * self.w/self.g) / (self.rho * self.Sw * self.cw)
        self.R_rho_y = (4 * self.w/self.g) / (self.rho * self.Sw * self.bw)
        self.R_gx = (self.g * self.cw) / (2 * self.V_0**2)
        self.R_gy = (self.g * self.bw) / (2 * self.V_0**2)
        self.R_xx = (8 * self.Ixxb) / (self.rho * self.Sw * self.bw**3)
        self.R_yy = (8 * self.Iyyb) / (self.rho * self.Sw * self.cw**3)
        self.R_zz = (8 * self.Izzb) / (self.rho * self.Sw * self.bw**3)
        self.R_xz = (8 * self.Ixzb) / (self.rho * self.Sw * self.bw**3)
        self.CY_beta = self.CS_beta
        self.CY_rbar = self.CS_rbar
        self.C_TV = 0
        self.CD_mu_hat = 0
        self.CL_mu_hat = 0
        self.Cm_mu_hat = 0
        self.CD_alpha_hat = 0
        # self.check()

        # order of the longitudinal analysis functions to run the code
        self.get_long_matrices()  # matrices for special longitudinal eigenproblem
        self.get_eigen_long()  # solver for the special longitudinal eigenproblem
        self.sort_long()  # sorts the longitudinal eigenvalues and eigenvectors in ascending order
        self.long_split(self.long_eigvals, self.long_eigvecs)  # splits the real and imaginary components
        self.get_long_amp()  # amplitude of each component of the eigenvectors
        self.get_long_phase_angle()  # phase angle of each component of the eigenvectors
        self.short_period()
        self.print_short_period()
        self.phugoid()
        self.print_phugoid()

        # order of the lateral analysis functions to run the code
        self.get_lat_matrices()  # matrices for special lateral eigenproblem
        self.get_eigen_lat()  # solver for the special lateral eigenproblem
        self.sort_lat()  # sorts the lateral eigenvalues and eigenvectors in ascending order
        self.lat_split(self.lat_eigvals, self.lat_eigvecs)  # splits the real and imaginary components
        self.get_lat_amp()  # amplitude of each component of the eigenvectors
        self.get_lat_phase_angle()  # phase angle of each component of the eigenvectors
        self.real_lat_eigen()
        self.roll_mode()
        self.print_roll_mode()
        self.spiral_mode()
        self.print_spiral_mode()
        self.dutch_roll_mode()
        self.print_dutch_roll()

        # approximation analysis values
        self.short_period_approx()
        self.phugoid_approx()
        self.roll_mode_approx()
        self.spiral_mode_approx()
        self.dutch_roll_approx()

    def check(self):
        print(self.CD_o)
        print(self.CD_alpha)
        print(self.V_0)
        print(self.R_rho_x)
        print(self.R_gx)
        print(self.R_yy)

    def get_long_matrices(self):
        # longitudinal A and B matrices for the special eigenproblem
        self.A_long = np.asarray([[(-2*self.CD_o + self.C_TV*np.cos(self.alpha_t0)), (self.CL_0 - self.CD_alpha), -self.CD_qbar, 0, 0, (-self.R_rho_x * self.R_gx * np.cos(self.theta0))],
                         [(-2*self.CL_0 - self.C_TV*np.sin(self.alpha_t0)), (-self.CL_alpha - self.CD_o), (-self.CL_qbar + self.R_rho_x), 0, 0, (-self.R_rho_x * self.R_gx * np.sin(self.theta0))],
                         [(2*self.Cm_0 + (self.C_TV*self.zt0)/self.cw), self.Cm_alpha, self.Cm_qbar, 0, 0, 0],
                         [np.cos(self.theta0), np.sin(self.theta0), 0, 0, 0, -np.sin(self.theta0)],
                         [-np.sin(self.theta0), np.cos(self.theta0), 0, 0, 0, -np.cos(self.theta0)],
                         [0, 0, 1, 0, 0, 0]])

        self.B_long = np.asarray([[(self.R_rho_x + self.CD_mu_hat), self.CD_alpha_hat, 0, 0, 0, 0],
                             [self.CL_mu_hat, (self.R_rho_x + self.CL_alpha_hat), 0, 0, 0, 0],
                             [-self.Cm_mu_hat, -self.Cm_alpha_hat, self.R_yy, 0, 0, 0],
                             [0, 0, 0, 1, 0, 0],
                             [0, 0, 0, 0, 1, 0],
                             [0, 0, 0, 0, 0, 1]])

    def get_eigen_long(self):
        # calculating the longitudinal eigenvalues and eigenvectors
        self.C_long = np.matmul(np.linalg.inv(self.B_long), self.A_long)

        eigvals, eigvecs = eig(self.C_long)
        # eigvals, eigvecs = eig(self.A_long, self.B_long)

        self.long_eigvals1 = np.array(eigvals)
        self.long_eigvecs1 = np.array(eigvecs)

        print('Longitudinal A-Matrix')  # printing out the longitudinal A matrix
        print('\n'.join([''.join(['{:>18.12f}'.format(item) for item in row])
                         for row in self.A_long]))

        print('\n')
        print('Longitudinal B-Matrix')  # printing out the longitudinal B matrix
        print('\n'.join([''.join(['{:>18.12f}'.format(item) for item in row])
                         for row in self.B_long]))

        print('\n')
        print('Longitudinal C-Matrix')  # printing out the longitudinal C matrix
        print('\n'.join([''.join(['{:>18.12f}'.format(item) for item in row])
                         for row in self.C_long]))
        print('\n')

        # print('\nLongitudinal Eigenvalues')  # printing out the longitudinal eigenvalues before sorting
        # print('\n'.join('{:>18.6f}'.format(item) for item in eigvals))
        #
        # print('\nLongitudinal Eigenvectors')  # printing out the longitudinal eigenvectors before sorting
        # print('\n'.join([''.join(['{:>18.6f}'.format(item) for item in row])
        #                  for row in eigvecs]))

    def long_split(self, eigen, eigvec):
        # getting the real and imaginary components of each eigenvalue
        self.long_real = eigen.real
        self.long_imag = eigen.imag
        self.long_eigvec_real = eigvec.real
        self.long_eigvec_imag = eigvec.imag

    def sort_long(self):
        # sorting the longitudinal eigenvalues and eigenvectors
        # sorting the lateral eigenvalues and eigenvectors
        idx = np.argsort(np.abs(self.long_eigvals1))
        self.long_eigvals = self.long_eigvals1[idx]
        self.long_eigvecs = self.long_eigvecs1[:, idx]

        print('\nLongitudinal Eigenvalues')  # printing out the longitudinal eigenvalues
        print('\n'.join('{:>18.6f}'.format(item) for item in self.long_eigvals))

        print('\nLongitudinal Eigenvectors')  # printing out the longitudinal eigenvectors
        print('\n'.join([''.join(['{:>18.6f}'.format(item) for item in row])
                         for row in self.long_eigvecs]))
        print('\n')

    def get_long_amp(self):
        # amplitude of each component of the eigenvectors
        self.long_amp = abs(self.long_eigvecs)

        # print('\nAmplitudes')  # printing out the amplitude of each component the eigenvectors
        # print('\n'.join([''.join(['{:>18.12f}'.format(item) for item in row])
        #                  for row in self.long_amp]))

    def get_long_phase_angle(self):
        # phase angle of each component of the eigenvectors
        self.long_pha = np.zeros((len(self.long_eigvecs), len(self.long_eigvecs)))
        for i in range(len(self.long_eigvecs)):
            for j in range(len(self.long_eigvecs)):
                self.long_pha[i, j] = (180/np.pi) * (np.arctan2(self.long_eigvec_imag[i, j], self.long_eigvec_real[i, j]))

        # print('\nPhase Angle')  # printing out the phase angle of each component of the eigenvectors
        # print('\n'.join([''.join(['{:>18.12f}'.format(item) for item in row])
        #                  for row in self.long_pha]))

    def short_period(self):
        # short period values
        # sorting the short-period longitudinal mode
        self.short_period1 = self.long_eigvals[5]
        self.short_period2 = self.long_eigvals[4]

        # damping rate [1/sec]
        self.short_damp1 = -self.short_period1.real * ((2 * self.V_0) / self.cw)
        self.short_damp2 = -self.short_period2.real * ((2 * self.V_0) / self.cw)

        # 99% damping time [sec] if convergent
        if self.short_period1.real < 0:
            self.short_99_1 = -np.log(0.01) / self.short_damp1
        if self.short_period1.real > 0:
            self.short_99_1 = 0
        if self.short_period2.real < 0:
            self.short_99_2 = -np.log(0.01) / self.short_damp2
        if self.short_period2.real > 0:
            self.short_99_2 = 0

        # doubling time [sec] if divergent
        if self.short_period1.real < 0:
            self.short_double1 = 0
        if self.short_period1.real > 0:
            self.short_double1 = -np.log(2) / self.short_damp1
        if self.short_period2.real < 0:
            self.short_double2 = 0
        if self.short_period2.real > 0:
            self.short_double2 = -np.log(2) / self.short_damp2

        # damped frequency [rad/s]
        if abs(self.short_period1.imag) > 0:
            self.short_damp_freq1 = abs(self.short_period1.imag) * ((2 * self.V_0) / self.cw)
        else:
            self.short_damp_freq1 = 0
        if abs(self.short_period2.imag) > 0:
            self.short_damp_freq2 = abs(self.short_period2.imag) * ((2 * self.V_0) / self.cw)
        else:
            self.short_damp_freq2 = 0

        # period [sec]
        if abs(self.short_period1.imag) > 0:
            self.short_per1 = (2 * np.pi) / self.short_damp_freq1
        else:
            self.short_per1 = 0
        if abs(self.short_period2.imag) > 0:
            self.short_per2 = (2 * np.pi) / self.short_damp_freq2
        else:
            self.short_per2 = 0

    def print_short_period(self):
        print("                                         Short Period 1                                            ")
        print("---------------------------------------------------------------------------------------------------")
        # short period 1 values
        print("Short Period Dimensionless Eigenvalue 1: ", self.short_period1)
        # Convergent or divergent
        if self.short_period1.real < 0:
            print("This mode is Convergent")
        if self.short_period1.real > 0:
            print("This mode is Divergent")
        # damping rate
        print("Damping Rate [1/sec]: ", self.short_damp1)
        # 99% damping time
        print("99% Damping Time [sec]: ", self.short_99_1 if self.short_99_1 != 0 else "N/A")
        # doubling time
        print("Doubling Time [sec]: ", self.short_double1 if self.short_double1 != 0 else "N/A")
        # damped natural frequency
        print("Damped Natural Frequency [rad/s]: ", self.short_damp_freq1 if self.short_damp_freq1 != 0 else "N/A")
        # period
        print("Period: ", self.short_per1 if self.short_per1 != 0 else "N/A")

        # matrix to show eigenvectors, phase angles, and amplitudes nicely
        eigenvector = np.zeros((6, 5), dtype=object)
        for i in range(len(self.long_eigvals)):
            eigenvector[i, 1] = self.long_eigvecs.real[5, i]
            eigenvector[i, 2] = self.long_eigvecs.imag[5, i]
            eigenvector[i, 3] = self.long_amp[5, i]
            eigenvector[i, 4] = self.long_pha[5, i]
            eigenvector[0, 0] = "\u0394u"  # delta mu
            eigenvector[1, 0] = "\u0394a"  # delta alpha
            eigenvector[2, 0] = "\u0394q"  # delta q
            eigenvector[3, 0] = "\u0394\u03B5x"  # delta epsilon x
            eigenvector[4, 0] = "\u0394\u03B5z"  # delta epsilon z
            eigenvector[5, 0] = "\u0394\u03F4x"  # delta theta

        print(' Eigenvector       Real                 Imaginary            Amplitude         Phase Angle [deg]')  # printing out the longitudinal eigenvectors
        print('\n'.join(['   '.join(['{:>12}'.format(item) if type(item) == str else '{:>18.12f}'.format(item) for item in row]) for row in eigenvector]))
        print('\n')

        # short period 2 values
        print("                                         Short Period 2                                            ")
        print("---------------------------------------------------------------------------------------------------")
        print("Short Period Dimensionless Eigenvalue 2: ", self.short_period2)
        # Convergent or divergent
        if self.short_period2.real < 0:
            print("This mode is Convergent")
        if self.short_period2.real > 0:
            print("This mode is Divergent")
        # damping rate
        print("Damping Rate [1/sec]: ", self.short_damp2)
        # 99% damping time
        print("99% Damping Time [sec]: ", self.short_99_2 if self.short_99_2 != 0 else "N/A")
        # doubling time
        print("Doubling Time [sec]: ", self.short_double2 if self.short_double2 != 0 else "N/A")
        # damped natural frequency
        print("Damped Natural Frequency [rad/s]: ", self.short_damp_freq2 if self.short_damp_freq2 != 0 else "N/A")
        # period
        print("Period: ", self.short_per2 if self.short_per2 != 0 else "N/A")

        # matrix to show eigenvectors, phase angles, and amplitudes nicely
        eigenvector2 = np.zeros((6, 5), dtype=object)
        for i in range(len(self.long_eigvals)):
            eigenvector2[i, 1] = self.long_eigvecs.real[4, i]
            eigenvector2[i, 2] = self.long_eigvecs.imag[4, i]
            eigenvector2[i, 3] = self.long_amp[4, i]
            eigenvector2[i, 4] = self.long_pha[4, i]
            eigenvector2[0, 0] = "\u0394u"  # delta mu
            eigenvector2[1, 0] = "\u0394a"  # delta alpha
            eigenvector2[2, 0] = "\u0394q"  # delta q
            eigenvector2[3, 0] = "\u0394\u03B5x"  # delta epsilon x
            eigenvector2[4, 0] = "\u0394\u03B5z"  # delta epsilon z
            eigenvector2[5, 0] = "\u0394\u03F4x"  # delta theta

        print(
            ' Eigenvector       Real                 Imaginary            Amplitude         Phase Angle [deg]')  # printing out the longitudinal eigenvectors
        print('\n'.join(
            ['   '.join(['{:>12}'.format(item) if type(item) == str else '{:>18.12f}'.format(item) for item in row]) for
             row in eigenvector2]))
        print('\n')

    def short_period_approx(self):
        # short period approximation values and error
        self.A_sp = self.R_yy * (self.R_rho_x + self.CL_alpha_hat)
        self.B_sp = (self.R_yy * (self.CL_alpha + self.CD_o)) - (self.Cm_qbar * (self.R_rho_x + self.CL_alpha_hat)) - (self.Cm_alpha_hat * (self.R_rho_x - self.CL_qbar))
        self.C_sp = (-self.Cm_qbar * (self.CL_alpha + self.CD_o)) - (self.Cm_alpha * (self.R_rho_x - self.CL_qbar))

        # calculating the eigenvalues
        self.sp_eigen1 = (-self.B_sp + np.sqrt((self.B_sp**2) - (4 * self.A_sp * self.C_sp))) / (2 * self.A_sp)
        self.sp_eigen2 = (-self.B_sp - np.sqrt((self.B_sp**2) - (4 * self.A_sp * self.C_sp))) / (2 * self.A_sp)

        # approximate damping rate [1/sec]
        if abs(self.sp_eigen1) > abs(self.sp_eigen2):
            self.sp_damp_approx = -(2*self.V_0/self.cw) * self.sp_eigen1.real
        if abs(self.sp_eigen2) > abs(self.sp_eigen1):
            self.sp_damp_approx = -(2 * self.V_0 / self.cw) * self.sp_eigen2.real

        # approximate 99% damping time [sec]
        self.sp_99_approx = -np.log(0.01) / self.sp_damp_approx

        # damped frequency [rad/s]
        if abs(self.sp_eigen1) > abs(self.sp_eigen2):
            self.sp_freq_approx = (2*self.V_0/self.cw) * abs(self.sp_eigen1.imag)
        if abs(self.sp_eigen2) > abs(self.sp_eigen1):
            self.sp_freq_approx = (2*self.V_0/self.cw) * abs(self.sp_eigen2.imag)

        # period [sec]
        if self.sp_freq_approx != 0:
            self.sp_per_approx = (2 * np.pi) / self.sp_freq_approx
        else:
            self.sp_per_approx = 0

        # percent error between approximate values and real values
        self.sp_damp_error = abs((self.sp_damp_approx - self.short_damp1) / self.short_damp1) * 100
        self.sp_99_error = abs((self.sp_99_approx - self.short_99_1) / self.short_99_1) * 100
        if self.short_damp_freq1 == 0:
            self.sp_freq_error = 0
        else:
            self.sp_freq_error = abs((self.sp_freq_approx - self.short_damp_freq1) / self.short_damp_freq1) * 100
        if self.short_per1 == 0:
            self.sp_per_error = 0
        else:
            self.sp_per_error = abs((self.sp_per_approx - self.short_per1) / self.short_per1) * 100

    def phugoid(self):
        # phugoid values
        # sorting the phugoid eigenvalues
        self.phugoid1 = self.long_eigvals[3]
        self.phugoid2 = self.long_eigvals[2]

        # damping ratio
        self.phugoid_damp_ratio = -(self.phugoid1 + self.phugoid2) / (2 * np.sqrt(self.phugoid1 * self.phugoid2))

        # undamped natural frequency
        self.phugoid_undamp_freq = ((2 * self.V_0)/self.cw) * np.sqrt(self.phugoid1 * self.phugoid2)

        # damping rate [1/sec]
        self.phugoid_damp1 = -self.phugoid1.real * ((2 * self.V_0) / self.cw)
        self.phugoid_damp2 = -self.phugoid2.real * ((2 * self.V_0) / self.cw)

        # 99% damping time [sec] if convergent
        if self.phugoid1.real < 0:
            self.phugoid_99_1 = -np.log(0.01) / self.phugoid_damp1
        if self.phugoid1.real > 0:
            self.phugoid_99_1 = 0
        if self.phugoid2.real < 0:
            self.phugoid_99_2 = -np.log(0.01) / self.phugoid_damp2
        if self.phugoid2.real > 0:
            self.phugoid_99_2 = 0

        # doubling time [sec] if divergent
        if self.phugoid1.real < 0:
            self.phugoid_double1 = 0
        if self.phugoid1.real.real > 0:
            self.phugoid_double1 = -np.log(2) / self.phugoid_damp1
        if self.phugoid2.real < 0:
            self.phugoid_double2 = 0
        if self.phugoid2.real > 0:
            self.phugoid_double2 = -np.log(2) / self.phugoid_damp2

        # damped frequency [rad/s]
        if abs(self.phugoid1.imag) > 0:
            self.phugoid_freq1 = abs(self.phugoid1.imag) * ((2 * self.V_0) / self.cw)
        else:
            self.phugoid_freq1 = 0
        if abs(self.phugoid2.imag) > 0:
            self.phugoid_freq2 = abs(self.phugoid2.imag) * ((2 * self.V_0) / self.cw)
        else:
            self.phugoid_freq2 = 0

        # period [sec]
        if abs(self.phugoid1.imag) > 0:
            self.phugoid_per1 = (2 * np.pi) / self.phugoid_freq1
        else:
            self.phugoid_per1 = 0
        if abs(self.phugoid2.imag) > 0:
            self.phugoid_per2 = (2 * np.pi) / self.phugoid_freq2
        else:
            self.phugoid_per2 = 0

    def print_phugoid(self):
        print("                                           Phugoid 1                                               ")
        print("---------------------------------------------------------------------------------------------------")
        # phugoid 1 values
        print("Phugoid Dimensionless Eigenvalue 1: ", self.phugoid1)
        # Convergent or divergent
        if self.phugoid1.real < 0:
            print("This mode is Convergent")
        if self.phugoid1.real > 0:
            print("This mode is Divergent")
        # damping rate
        print("Damping Rate [1/sec]: ", self.phugoid_damp1)
        # 99% damping time
        print("99% Damping Time [sec]: ", self.phugoid_99_1 if self.phugoid_99_1 != 0 else "N/A")
        # doubling time
        print("Doubling Time [sec]: ", self.phugoid_double1 if self.phugoid_double1 != 0 else "N/A")
        # damped natural frequency
        print("Damped Natural Frequency [rad/s]: ", self.phugoid_freq1 if self.phugoid_freq1 != 0 else "N/A")
        # period
        print("Period: ", self.phugoid_per1 if self.phugoid_per1 != 0 else "N/A")
        # damping ratio (because these eigenvalues are complex)
        print("Damping Ratio: ", self.phugoid_damp_ratio.real)
        # undamped natural frequency
        print("Undamped Natural Frequency [rad/s]: ", self.phugoid_undamp_freq.real)

        # matrix to show eigenvectors, phase angles, and amplitudes nicely
        eigenvector = np.zeros((6, 5), dtype=object)
        for i in range(len(self.long_eigvals)):
            eigenvector[i, 1] = self.long_eigvecs.real[3, i]
            eigenvector[i, 2] = self.long_eigvecs.imag[3, i]
            eigenvector[i, 3] = self.long_amp[3, i]
            eigenvector[i, 4] = self.long_pha[3, i]
            eigenvector[0, 0] = "\u0394u"  # delta mu
            eigenvector[1, 0] = "\u0394a"  # delta alpha
            eigenvector[2, 0] = "\u0394q"  # delta q
            eigenvector[3, 0] = "\u0394\u03B5x"  # delta epsilon x
            eigenvector[4, 0] = "\u0394\u03B5z"  # delta epsilon z
            eigenvector[5, 0] = "\u0394\u03F4x"  # delta theta

        print(
            ' Eigenvector       Real                 Imaginary            Amplitude         Phase Angle [deg]')  # printing out the longitudinal eigenvectors
        print('\n'.join(
            ['   '.join(['{:>12}'.format(item) if type(item) == str else '{:>18.12f}'.format(item) for item in row]) for
             row in eigenvector]))
        print('\n')

        # phugoid 2 values
        print("                                           Phugoid 2                                               ")
        print("---------------------------------------------------------------------------------------------------")
        print("Phugoid Dimensionless Eigenvalue 2: ", self.phugoid2)
        # Convergent or divergent
        if self.phugoid2.real < 0:
            print("This mode is Convergent")
        if self.phugoid2.real > 0:
            print("This mode is Divergent")
        # damping rate
        print("Damping Rate [1/sec]: ", self.phugoid_damp2)
        # 99% damping time
        print("99% Damping Time [sec]: ", self.phugoid_99_2 if self.phugoid_99_2 != 0 else "N/A")
        # doubling time
        print("Doubling Time [sec]: ", self.phugoid_double2 if self.phugoid_double2 != 0 else "N/A")
        # damped natural frequency
        print("Damped Natural Frequency [rad/s]: ", self.phugoid_freq2 if self.phugoid_freq2 != 0 else "N/A")
        # period
        print("Period: ", self.phugoid_per2 if self.phugoid_per2 != 0 else "N/A")
        # damping ratio (because these eigenvalues are complex)
        print("Damping Ratio: ", self.phugoid_damp_ratio.real)
        # undamped natural frequency
        print("Undamped Natural Frequency: ", self.phugoid_undamp_freq.real)

        # matrix to show eigenvectors, phase angles, and amplitudes nicely
        eigenvector2 = np.zeros((6, 5), dtype=object)
        for i in range(len(self.long_eigvals)):
            eigenvector2[i, 1] = self.long_eigvecs.real[2, i]
            eigenvector2[i, 2] = self.long_eigvecs.imag[2, i]
            eigenvector2[i, 3] = self.long_amp[2, i]
            eigenvector2[i, 4] = self.long_pha[2, i]
            eigenvector2[0, 0] = "\u0394u"  # delta mu
            eigenvector2[1, 0] = "\u0394a"  # delta alpha
            eigenvector2[2, 0] = "\u0394q"  # delta q
            eigenvector2[3, 0] = "\u0394\u03B5x"  # delta epsilon x
            eigenvector2[4, 0] = "\u0394\u03B5z"  # delta epsilon z
            eigenvector2[5, 0] = "\u0394\u03F4x"  # delta theta

        print(
            ' Eigenvector       Real                 Imaginary            Amplitude         Phase Angle [deg]')  # printing out the longitudinal eigenvectors
        print('\n'.join(
            ['   '.join(['{:>12}'.format(item) if type(item) == str else '{:>18.12f}'.format(item) for item in row]) for
             row in eigenvector2]))
        print("---------------------------------------------------------------------------------------------------")
        print('\n')

    def phugoid_approx(self):
        # phugoid approximation values and error
        # phugoid drag damping (sigma_D) 11.32
        self.phugoid_drag = (self.g / self.V_0) * (self.CD_o / self.CL_0)

        # phugoid pitch damping (sigma_q) 11.33
        self.phugoid_pitch = (self.g / self.V_0) * (((self.CL_0 - self.CD_alpha) * self.Cm_qbar) / ((self.R_rho_x * self.Cm_alpha) + ((self.CD_o + self.CL_alpha) * self.Cm_qbar)))

        # phugoid stability ratio 11.35
        self.phugoid_rps = (self.R_rho_x * self.Cm_alpha) / ((self.R_rho_x * self.Cm_alpha) + ((self.CD_o + self.CL_alpha) * self.Cm_qbar))

        # phugoid phase damping 11.34
        self.phugoid_phase = (-self.g / self.V_0) * self.R_gx * self.phugoid_rps * (((self.R_rho_x * self.Cm_qbar) - (self.R_yy * (self.CD_o + self.CL_alpha))) / ((self.R_rho_x * self.Cm_alpha) + ((self.CD_o + self.CL_alpha) * self.Cm_qbar)))

        # phugoid damping rate approximation
        self.phugoid_damp_approx = self.phugoid_drag + self.phugoid_pitch + self.phugoid_phase

        # 99% damping time approximation
        self.phugoid_99_approx = -np.log(0.01) / self.phugoid_damp_approx

        # phugoid damped frequency approximation
        self.phugoid_freq_approx = np.sqrt((2 * (self.g/self.V_0)**2 * self.phugoid_rps) - (self.phugoid_drag + self.phugoid_pitch)**2)

        # phugoid period approximation
        self.phugoid_per_approx = (2 * np.pi) / self.phugoid_freq_approx

        # percent error between approximate values and real values
        self.phugoid_damp_error = abs((self.phugoid_damp_approx - self.phugoid_damp1) / self.phugoid_damp1) * 100
        self.phugoid_99_error = abs((self.phugoid_99_approx - self.phugoid_99_1) / self.phugoid_99_1) * 100
        if self.phugoid_freq1 == 0:
            self.phugoid_freq_error = 0
        else:
            self.phugoid_freq_error = abs((self.phugoid_freq_approx - self.phugoid_freq1) / self.phugoid_freq1) * 100
        if self.phugoid_per1 == 0:
            self.phugoid_per_error = 0
        else:
            self.phugoid_per_error = abs((self.phugoid_per_approx - self.phugoid_per1) / self.phugoid_per1) * 100

    def get_lat_matrices(self):
        self.A_lat = np.asarray([[self.CY_beta, self.CY_pbar, (self.CY_rbar - self.R_rho_y), 0, (self.R_rho_y * self.R_gy * np.cos(self.theta0)), 0],
                                 [self.Cl_beta, self.Cl_pbar, self.Cl_rbar, 0, 0, 0],
                                 [self.Cn_beta, self.Cn_pbar, self.Cn_rbar, 0, 0, 0],
                                 [1, 0, 0, 0, 0, np.cos(self.theta0)],
                                 [0, 1, np.tan(self.theta0), 0, 0, 0],
                                 [0, 0, 1/np.cos(self.theta0), 0, 0, 0]])

        self.B_lat = np.asarray([[self.R_rho_y, 0, 0, 0, 0, 0],
                                 [0, self.R_xx, -self.R_xz, 0, 0, 0],
                                 [0, -self.R_xz, self.R_zz, 0, 0, 0],
                                 [0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 0, 1]])

    def get_eigen_lat(self):
        # calculating the lateral eigenvalues and eigenvectors
        self.C_lat = np.matmul(np.linalg.inv(self.B_lat), self.A_lat)

        eigvals, eigvecs = eig(self.C_lat)
        # eigvals, eigvecs = eig(self.A_long, self.B_long)

        self.lat_eigvals1 = np.array(eigvals)
        self.lat_eigvecs1 = np.array(eigvecs)

        print('Lateral A-Matrix')  # printing out the longitudinal A matrix
        print('\n'.join([''.join(['{:>18.12f}'.format(item) for item in row])
                         for row in self.A_lat]))

        print('\n')
        print('Lateral B-Matrix')  # printing out the longitudinal B matrix
        print('\n'.join([''.join(['{:>18.12f}'.format(item) for item in row])
                         for row in self.B_lat]))

        print('\n')
        print('Lateral C-Matrix')  # printing out the longitudinal C matrix
        print('\n'.join([''.join(['{:>18.12f}'.format(item) for item in row])
                         for row in self.C_lat]))
        print('\n')

        # print('\nLateral Eigenvalues')  # printing out the longitudinal eigenvalues before sorting
        # print('\n'.join('{:>18.6f}'.format(item) for item in eigvals))
        #
        # print('\nLateral Eigenvectors')  # printing out the longitudinal eigenvectors before sorting
        # print('\n'.join([''.join(['{:>18.6f}'.format(item) for item in row])
        #                  for row in eigvecs]))

    def lat_split(self, eigen, eigvec):
        # getting the real and imaginary components of each eigenvalue
        self.lat_real = eigen.real
        self.lat_imag = eigen.imag
        self.lat_eigvec_real = eigvec.real
        self.lat_eigvec_imag = eigvec.imag

    def sort_lat(self):
        # sorting the longitudinal eigenvalues and eigenvectors
        # sorting the lateral eigenvalues and eigenvectors
        idx = np.argsort(np.abs(self.lat_eigvals1))
        self.lat_eigvals = self.lat_eigvals1[idx]
        self.lat_eigvecs = self.lat_eigvecs1[:, idx]

        print('\nLateral Eigenvalues')  # printing out the lateral eigenvalues
        print('\n'.join('{:>18.6f}'.format(item) for item in self.lat_eigvals))

        print('\nLateral Eigenvectors')  # printing out the lateral eigenvectors
        print('\n'.join([''.join(['{:>18.6f}'.format(item) for item in row])
                         for row in self.lat_eigvecs]))
        print('\n')

    def get_lat_amp(self):
        # amplitude of each component of the eigenvectors
        self.lat_amp = abs(self.lat_eigvecs)

        # print('\nAmplitudes')  # printing out the amplitude of each component the eigenvectors
        # print('\n'.join([''.join(['{:>18.12f}'.format(item) for item in row])
        #                  for row in self.lat_amp]))

    def get_lat_phase_angle(self):
        # phase angle of each component of the eigenvectors
        self.lat_pha = np.zeros((len(self.lat_eigvecs), len(self.lat_eigvecs)))
        for i in range(len(self.lat_eigvecs)):
            for j in range(len(self.lat_eigvecs)):
                self.lat_pha[i, j] = (180 / np.pi) * (
                    np.arctan2(self.lat_eigvec_imag[i, j], self.lat_eigvec_real[i, j]))

        # print('\nPhase Angle')  # printing out the phase angle of each component of the eigenvectors
        # print('\n'.join([''.join(['{:>18.12f}'.format(item) for item in row])
        #                  for row in self.lat_pha]))

    def real_lat_eigen(self):
        # separating the real eigenvalues from the complex eigenvalues
        self.real_lat = []
        self.complex_lat = []

        for i in range(len(self.lat_eigvals)):
            if self.lat_eigvals[i].imag == 0:
                self.real_lat.append(self.lat_eigvals[i])
            else:
                self.complex_lat.append(self.lat_eigvals[i])

    def roll_mode(self):
        # roll mode values
        # separating the roll mode eigenvalue
        self.roll_mode = self.real_lat[3]

        # damping rate [1/sec]
        self.roll_damp = -self.roll_mode.real * ((2 * self.V_0) / self.bw)

        # 99% damping time [sec] if convergent
        if self.roll_mode.real < 0:
            self.roll_99 = -np.log(0.01) / self.roll_damp
        if self.roll_mode.real > 0:
            self.roll_99 = 0

        # doubling time [sec] if divergent
        if self.roll_mode.real < 0:
            self.roll_double = 0
        if self.roll_mode.real > 0:
            self.roll_double = -np.log(2) / self.roll_damp

        # damped frequency [rad/s]
        if abs(self.roll_mode.imag) > 0:
            self.roll_freq = abs(self.roll_mode.imag) * ((2 * self.V_0) / self.bw)
        else:
            self.roll_freq = 0

        # period [sec]
        if abs(self.roll_mode.imag) > 0:
            self.roll_per = (2 * np.pi) / self.roll_freq
        else:
            self.roll_per = 0

    def print_roll_mode(self):
        print("                                           Roll Mode                                               ")
        print("---------------------------------------------------------------------------------------------------")
        # roll mode values
        print("Roll Mode Dimensionless Eigenvalue: ", self.roll_mode)
        # Convergent or divergent
        if self.roll_mode.real < 0:
            print("This mode is Convergent")
        if self.roll_mode.real > 0:
            print("This mode is Divergent")
        # damping rate
        print("Damping Rate [1/sec]: ", self.roll_damp)
        # 99% damping time
        print("99% Damping Time [sec]: ", self.roll_99 if self.roll_99 != 0 else "N/A")
        # doubling time
        print("Doubling Time [sec]: ", self.roll_double if self.roll_double != 0 else "N/A")
        # damped natural frequency
        print("Damped Natural Frequency [rad/s]: ", self.roll_freq if self.roll_freq != 0 else "N/A")
        # period
        print("Period: ", self.roll_per if self.roll_per != 0 else "N/A")

        # matrix to show eigenvectors, phase angles, and amplitudes nicely
        eigenvector = np.zeros((6, 5), dtype=object)
        for i in range(len(self.long_eigvals)):
            eigenvector[i, 1] = self.lat_eigvecs.real[5, i]
            eigenvector[i, 2] = self.lat_eigvecs.imag[5, i]
            eigenvector[i, 3] = self.lat_amp[5, i]
            eigenvector[i, 4] = self.lat_pha[5, i]
            eigenvector[0, 0] = "\u0394u"  # delta mu
            eigenvector[1, 0] = "\u0394a"  # delta alpha
            eigenvector[2, 0] = "\u0394q"  # delta q
            eigenvector[3, 0] = "\u0394\u03B5x"  # delta epsilon x
            eigenvector[4, 0] = "\u0394\u03B5z"  # delta epsilon z
            eigenvector[5, 0] = "\u0394\u03F4x"  # delta theta

        print(
            ' Eigenvector       Real                 Imaginary            Amplitude         Phase Angle [deg]')  # printing out the lateral eigenvectors
        print('\n'.join(
            ['   '.join(['{:>12}'.format(item) if type(item) == str else '{:>18.12f}'.format(item) for item in row]) for
             row in eigenvector]))
        print('\n')

    def roll_mode_approx(self):
        # roll mode approximation values
        # approximate roll mode eigenvalue
        self.roll_approx = self.Cl_pbar / self.R_xx

        # roll mode damping rate approximation
        self.roll_damp_approx = - ((self.rho * self.Sw * self.bw**2 * self.V_0) / (4 * self.Ixxb)) * self.Cl_pbar

        # roll mode 99% damping time approximation
        self.roll_99_approx = -np.log(0.01) / self.roll_damp_approx

        # error between real and approximate values
        self.roll_damp_error = abs((self.roll_damp_approx - self.roll_damp) / self.roll_damp) * 100
        self.roll_99_error = abs((self.roll_99_approx - self.roll_99) / self.roll_99) * 100

    def spiral_mode(self):
        # spiral mode values
        # separating the spiral mode eigenvalue
        self.spiral_mode = self.real_lat[2]

        # damping rate [1/sec]
        self.spiral_damp = -self.spiral_mode.real * ((2 * self.V_0) / self.bw)

        # 99% damping time [sec] if convergent
        if self.spiral_mode.real < 0:
            self.spiral_99 = -np.log(0.01) / self.spiral_damp
        if self.spiral_mode.real > 0:
            self.spiral_99 = 0

        # doubling time [sec] if divergent
        if self.spiral_mode.real < 0:
            self.spiral_double = 0
        if self.spiral_mode.real > 0:
            self.spiral_double = -np.log(2) / self.spiral_damp

        # damped frequency [rad/s]
        if abs(self.spiral_mode.imag) > 0:
            self.spiral_freq = abs(self.spiral_mode.imag) * ((2 * self.V_0) / self.bw)
        else:
            self.spiral_freq = 0

        # period [sec]
        if abs(self.spiral_mode.imag) > 0:
            self.spiral_per = (2 * np.pi) / self.spiral_freq
        else:
            self.spiral_per = 0

    def print_spiral_mode(self):
        print("                                          Spiral Mode                                              ")
        print("---------------------------------------------------------------------------------------------------")
        # spiral mode values
        print("Spiral Mode Dimensionless Eigenvalue: ", self.spiral_mode)
        # Convergent or divergent
        if self.spiral_mode.real < 0:
            print("This mode is Convergent")
        if self.spiral_mode.real > 0:
            print("This mode is Divergent")
        # damping rate
        print("Damping Rate [1/sec]: ", self.spiral_damp)
        # 99% damping time
        print("99% Damping Time [sec]: ", self.spiral_99 if self.spiral_99 != 0 else "N/A")
        # doubling time
        print("Doubling Time [sec]: ", self.spiral_double if self.spiral_double != 0 else "N/A")
        # damped natural frequency
        print("Damped Natural Frequency [rad/s]: ", self.spiral_freq if self.spiral_freq != 0 else "N/A")
        # period
        print("Period: ", self.spiral_per if self.spiral_per != 0 else "N/A")

        # matrix to show eigenvectors, phase angles, and amplitudes nicely
        # index value of this mode needs to be changed manually based on the location of the appropriate eigenvalue
        # just for printing this matrix, the code above will always print the correct values
        eigenvector = np.zeros((6, 5), dtype=object)
        for i in range(len(self.long_eigvals)):
            eigenvector[i, 1] = self.lat_eigvecs.real[2, i]
            eigenvector[i, 2] = self.lat_eigvecs.imag[2, i]
            eigenvector[i, 3] = self.lat_amp[2, i]
            eigenvector[i, 4] = self.lat_pha[2, i]
            eigenvector[0, 0] = "\u0394u"  # delta mu
            eigenvector[1, 0] = "\u0394a"  # delta alpha
            eigenvector[2, 0] = "\u0394q"  # delta q
            eigenvector[3, 0] = "\u0394\u03B5x"  # delta epsilon x
            eigenvector[4, 0] = "\u0394\u03B5z"  # delta epsilon z
            eigenvector[5, 0] = "\u0394\u03F4x"  # delta theta

        print(
            ' Eigenvector       Real                 Imaginary            Amplitude         Phase Angle [deg]')  # printing out the lateral eigenvectors
        print('\n'.join(
            ['   '.join(['{:>12}'.format(item) if type(item) == str else '{:>18.12f}'.format(item) for item in row]) for
             row in eigenvector]))
        print('\n')

    def spiral_mode_approx(self):
        # spiral mode approximation values
        self.spiral_approx = -((self.g * self.bw) / (2 * self.V_0**2)) * (((self.Cl_beta * self.Cn_rbar) - (self.Cl_rbar * self.Cn_beta)) / ((self.Cl_beta * self.Cn_pbar) - (self.Cl_pbar * self.Cn_beta)))

        # spiral mode damping rate approximation
        self.spiral_damp_approx = -((2 * self.V_0) / self.bw) * self.spiral_approx

        # spiral mode 99% damping time approximation [sec] if convergent
        if self.spiral_approx.real < 0:
            self.spiral_99_approx = -np.log(0.01) / self.spiral_damp_approx
        if self.spiral_approx.real > 0:
            self.spiral_99_approx = 0

        # spiral mode doubling time approximation [sec] if divergent
        if self.spiral_approx.real < 0:
            self.spiral_double_approx = 0
        if self.spiral_approx.real > 0:
            self.spiral_double_approx = -np.log(2) / self.spiral_damp_approx

        # percent error between approximate values and real values
        self.spiral_damp_error = abs((self.spiral_damp_approx - self.spiral_damp) / self.spiral_damp) * 100
        if self.spiral_99_approx == 0:
            self.spiral_99_error = 0
        else:
            self.spiral_99_error = abs((self.spiral_99_approx - self.spiral_99) / self.spiral_99) * 100
        if self.spiral_double_approx == 0:
            self.spiral_double_error = 0
        else:
            self.spiral_double_error = abs((self.spiral_double_approx - self.spiral_double) / self.spiral_double) * 100

    def dutch_roll_mode(self):
        # dutch roll mode values
        # separating the dutch roll mode eigenvalues
        self.dr_mode1 = self.complex_lat[1]
        self.dr_mode2 = self.complex_lat[0]

        # damping ratio
        self.dr_damp_ratio = -(self.dr_mode1 + self.dr_mode2) / (2 * np.sqrt(self.dr_mode1 * self.dr_mode2))

        # undamped natural frequency
        self.dr_undamp_freq = ((2 * self.V_0) / self.bw) * np.sqrt(self.dr_mode1 * self.dr_mode2)

        # damping rate [1/sec]
        self.dr_damp = -self.dr_mode1.real * ((2 * self.V_0) / self.bw)

        # 99% damping time [sec] if convergent
        if self.dr_mode1.real < 0:
            self.dr_99 = -np.log(0.01) / self.dr_damp
        if self.dr_mode1.real > 0:
            self.dr_99 = 0

        # doubling time [sec] if divergent
        if self.dr_mode1.real < 0:
            self.dr_double = 0
        if self.dr_mode1.real > 0:
            self.dr_double = -np.log(2) / self.dr_damp

        # damped frequency [rad/s]
        self.dr_freq = abs(self.dr_mode1.imag) * ((2 * self.V_0) / self.bw)

        # period [sec]
        self.dr_per = (2 * np.pi) / self.dr_freq

    def print_dutch_roll(self):
        print("                                       Dutch Roll Mode                                             ")
        print("---------------------------------------------------------------------------------------------------")
        # dutch roll mode values
        print("Dutch Roll Mode Dimensionless Eigenvalue 1: ", self.dr_mode1)
        print("Dutch Roll Mode Dimensionless Eigenvalue 2: ", self.dr_mode2)
        # Convergent or divergent
        if self.dr_mode1.real < 0:
            print("This mode is Convergent")
        if self.dr_mode1.real > 0:
            print("This mode is Divergent")
        # damping rate
        print("Damping Rate [1/sec]: ", self.dr_damp)
        # 99% damping time
        print("99% Damping Time [sec]: ", self.dr_99 if self.dr_99 != 0 else "N/A")
        # doubling time
        print("Doubling Time [sec]: ", self.dr_double if self.dr_double != 0 else "N/A")
        # damped natural frequency
        print("Damped Natural Frequency [rad/s]: ", self.dr_freq if self.dr_freq != 0 else "N/A")
        # period
        print("Period: ", self.dr_per if self.dr_per != 0 else "N/A")
        # damping ratio (because these eigenvalues are complex)
        print("Damping Ratio: ", self.dr_damp_ratio.real)
        # undamped natural frequency
        print("Undamped Natural Frequency [rad/s]: ", self.dr_undamp_freq.real)

        # matrix to show eigenvectors, phase angles, and amplitudes nicely
        # index value of this mode needs to be changed manually based on the location of the appropriate eigenvalue
        # just for printing this matrix, the code above will always print the correct values
        eigenvector = np.zeros((6, 5), dtype=object)
        for i in range(len(self.long_eigvals)):
            eigenvector[i, 1] = self.lat_eigvecs.real[4, i]
            eigenvector[i, 2] = self.lat_eigvecs.imag[4, i]
            eigenvector[i, 3] = self.lat_amp[4, i]
            eigenvector[i, 4] = self.lat_pha[4, i]
            eigenvector[0, 0] = "\u0394u"  # delta mu
            eigenvector[1, 0] = "\u0394a"  # delta alpha
            eigenvector[2, 0] = "\u0394q"  # delta q
            eigenvector[3, 0] = "\u0394\u03B5x"  # delta epsilon x
            eigenvector[4, 0] = "\u0394\u03B5z"  # delta epsilon z
            eigenvector[5, 0] = "\u0394\u03F4x"  # delta theta

        print(
            ' Eigenvector 1     Real                 Imaginary            Amplitude         Phase Angle [deg]')  # printing out the lateral eigenvectors
        print('\n'.join(
            ['   '.join(['{:>12}'.format(item) if type(item) == str else '{:>18.12f}'.format(item) for item in row]) for
             row in eigenvector]))
        print('\n')

        eigenvector2 = np.zeros((6, 5), dtype=object)
        for i in range(len(self.long_eigvals)):
            eigenvector2[i, 1] = self.lat_eigvecs.real[3, i]
            eigenvector2[i, 2] = self.lat_eigvecs.imag[3, i]
            eigenvector2[i, 3] = self.lat_amp[3, i]
            eigenvector2[i, 4] = self.lat_pha[3, i]
            eigenvector2[0, 0] = "\u0394u"  # delta mu
            eigenvector2[1, 0] = "\u0394a"  # delta alpha
            eigenvector2[2, 0] = "\u0394q"  # delta q
            eigenvector2[3, 0] = "\u0394\u03B5x"  # delta epsilon x
            eigenvector2[4, 0] = "\u0394\u03B5z"  # delta epsilon z
            eigenvector2[5, 0] = "\u0394\u03F4x"  # delta theta

        print(
            ' Eigenvector 2     Real               Imaginary            Amplitude         Phase Angle [deg]')  # printing out the lateral eigenvectors
        print('\n'.join(
            ['   '.join(['{:>12}'.format(item) if type(item) == str else '{:>18.12f}'.format(item) for item in row]) for
             row in eigenvector2]))
        print("---------------------------------------------------------------------------------------------------")
        print('\n')

    def dutch_roll_approx(self):
        # dutch roll approximation values
        # dutch roll stability ratio 11.49
        self.dutch_rds = ((self.Cl_beta * ((self.R_gy * self.R_rho_y * self.R_zz) - ((self.R_rho_y - self.CY_rbar) * self.Cn_pbar))) - (self.CY_beta * self.Cl_rbar * self.Cn_pbar)) / (self.R_rho_y * self.R_zz * self.Cl_pbar)

        # dutch roll damping rate approximation 11.47
        self.dr_damp_approx = -(self.V_0 / self.bw) * ((self.CY_beta / self.R_rho_y) + (self.Cn_rbar / self.R_zz) - ((self.Cl_rbar * self.Cn_pbar) / (self.Cl_pbar * self.R_zz)) + ((self.R_gy * ((self.Cl_rbar * self.Cn_beta) - (self.Cl_beta * self.Cn_rbar))) / (self.Cl_pbar * (self.Cn_beta + ((self.CY_beta * self.Cn_rbar) / self.R_rho_y)))) - (self.R_xx * (self.dutch_rds / self.Cl_pbar)))

        # dutch roll 99% damping frequency approximation
        self.dr_99_approx = -np.log(0.01) / self.dr_damp_approx

        # dutch roll damped frequency approximation
        self.dr_freq_approx = ((2 * self.V_0) / self.bw) * np.sqrt(((1 - (self.CY_rbar / self.R_rho_y)) * (self.Cn_beta / self.R_zz)) + ((self.CY_beta * self.Cn_rbar) / (self.R_rho_y * self.R_zz)) + self.dutch_rds - (0.25 * ((self.CY_beta / self.R_rho_y) + (self.Cn_rbar / self.R_zz))**2))

        # dutch roll period approximation
        self.dr_per_approx = (2 * np.pi) / self.dr_freq_approx

        # percent error between approximate values and real values
        self.dr_damp_error = abs((self.dr_damp_approx - self.dr_damp) / self.dr_damp) * 100
        self.dr_99_error = abs((self.dr_99_approx - self.dr_99) / self.dr_99) * 100
        if self.dr_freq == 0:
            self.dr_freq_error = 0
        else:
            self.dr_freq_error = abs((self.dr_freq_approx - self.dr_freq) / self.dr_freq) * 100
        if self.phugoid_per1 == 0:
            self.dr_per_error = 0
        else:
            self.dr_per_error = abs((self.dr_per_approx - self.dr_per) / self.dr_per) * 100

if __name__ == "__main__":
    Dynamic_mode_analysis("2305.json")
