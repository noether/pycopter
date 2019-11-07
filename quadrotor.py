import numpy as np
from scipy import linalg as la

class quadrotor:
    def __init__(self, tag, m, l, J, CDl, CDr, kt, km, kw, att, \
            pqr, xyz, v_ned, w):
        # physical constants
        self.tag = tag
        self.m = m   # [Kg]
        self.l = l   # [m]
        self.J = J   # Inertia matrix  [Kg/m^2]
        self.Jinv = la.inv(J)
        self.CDl = CDl  # Linear Drag coefficient
        self.CDr = CDr  # Angular Drag coefficient
        self.kt = kt    # Propeller thrust [N s^2]
        self.km = km    # Propeller moment [N m s^2]
        self.kw = kw    # Motor transient [1/s]

        # Configuration of the propellers
        self.w_to_Tlmn = np.array([[   -kt,  -kt,  -kt,   -kt],\
                                   [     0,-l*kt,    0,  l*kt],\
                                   [  l*kt,    0,-l*kt,     0],\
                                   [   -km,   km,  -km,   km]])
        self.Tlmn_to_w = la.inv(self.w_to_Tlmn)
        
        # Physical variables
        self.att = att # Attitude [rad]
        self.pqr = pqr # Body angular velocity [rad/sec]
        self.xyz = xyz # Body position NED [m]
        self.v_ned = v_ned # Body linear velocity NED [m/sec]
        self.w = w # Actual angular velocity of the propellers [rad/sec]
        self.Ft = np.array([0.0, 0.0, 0.0]) # Motor (Thrust) Force [N]
        self.Fa = np.array([0.0, 0.0, 0.0]) # Aerodynamic Forces [N]
        self.Mt = np.array([0.0, 0.0, 0.0]) # Motor Moment [N m]
        self.Ma = np.array([0.0, 0.0, 0.0]) # Aerodynamic Moments [N m]
        self.crashed = 0 # Ground hit?

        ## GNC variables
        # Geometric (1) or std (0) attitude controller
        self.att_con = 1

        # Gains for the attitude controller
        self.kp = 2
        self.kq = 2
        self.kr = 2
        # Gains for classical controllers
        self.k_pos = 2e-1
        self.k_vel = 2e-1
        # Gains for Lyapunov controllers
        self.k_alt = 1e-2
        self.k_vz = 1
        self.k_xy = 1e-2
        self.k_vxy = 1e-1
        self.k_xi_g_v = 1e-1
        self.k_xi_g_e_alt = 5e-3
        self.k_xi_CD_e_v = 1e-3
        self.e_alt = 0 # We need it for the estimator xi_g
        self.e_v = np.array([0, 0]) # We need it for the estimator xi_Cd
        # Gains for geometric att controller
        self.k_eR = 5e-3
        self.k_om = 1e-2

        self.T_d = 0 # Desired thrust [N]
        self.lmn_d = np.array([0.0, 0.0, 0.0]) # Desired angular momentum [N m]
        self.w_d = w # Desired angular velocity for the propellers [rad/sec]
        # Desired attitude roll, pitch [rads]
        self.att_d = np.array([0.0, 0.0, 0.0])
        # Hoovering desired 3D position NED [m]
        self.xyz_d = np.array([0.0, 0.0, 0.0])
        self.v_ned_d = np.array([0.0, 0.0, 0.0]) # Desired vel 3D NED
        self.yaw_d = 0  # Desired yaw [rad]
        # Estimators
        self.xi_g = 9.8 # Initial guess of gravity
        self.xi_CD = 0

    ### GNC Functions ###
    def control_att(self):
        # Attitude controller Lyapunov approach
        ephi = self.att[0] - self.att_d[0]
        ethe = self.att[1] - self.att_d[1]
        epsi = self.att[2] - self.att_d[2]

        # Desired moments
        self.lmn_d[0] = -self.J[0, 0]*(ephi + self.kp*self.pqr[0]) \
             -(self.J[1, 1]-self.J[2, 2])*self.pqr[1]*self.pqr[2]

        self.lmn_d[1] = -self.J[1, 1]*(ethe + self.kq*self.pqr[1]) \
             -(self.J[2, 2]-self.J[0, 0])*self.pqr[2]*self.pqr[0] \

        self.lmn_d[2] = -self.J[2, 2]*(epsi + self.kr*self.pqr[2])

    def control_att_geometric(self):
        R = self.Rot_bn().transpose()
        Rd = self.Rotd_bn(self.att_d[0], self.att_d[1], self.att_d[2]).transpose()

        e_RM = 0.5*(Rd.transpose().dot(R) - R.transpose().dot(Rd))
        e_R = self.build_vector_from_tensor(e_RM)

        om = np.array([self.pqr[0], self.pqr[1], self.pqr[2]])
        e_om = om

        M = -self.k_eR*e_R -self.k_om*e_om + np.cross(om, self.J.dot(om))

        self.lmn_d = M


    ## Lyapunov based controllers

    # Input: Desired 3D position
    def set_xyz_ned_lya(self, xyz_d):
        e_alt  = self.xyz[2] - xyz_d[2]
        self.e_alt = e_alt
        e_xy   = self.xyz[0:2] - xyz_d[0:2]
        
        self.T_d = (-self.xi_g -self.k_alt*e_alt \
                -self.k_vz*self.v_ned[2])*self.m

        axy = -self.k_xy*e_xy -self.k_vxy*self.v_ned[0:2]
        ax = axy[0]
        ay = axy[1]
        # Guidance attitude
        phi_d = -self.m/self.T_d*(ay*np.cos(self.att[2])-ax*np.sin(self.att[2]))
        the_d =  self.m/self.T_d*(ax*np.cos(self.att[2])+ay*np.sin(self.att[2]))

        # Control motors
        self.att_d = np.array([phi_d, the_d, self.yaw_d])
        if self.att_con == 0:
            self.control_att()
        elif self.att_con == 1:
            self.control_att_geometric()

        self.w_d = np.sqrt(self.Tlmn_to_w.dot(np.append(self.T_d, self.lmn_d)))

    # Input: Desired 2D acceleration, Desired altitude
    def set_a_2D_alt_lya(self, a_2d_d, altitude_d):
        e_alt  = self.xyz[2] - altitude_d
        self.e_alt = e_alt
        self.T_d = (-self.xi_g -self.k_alt*e_alt \
                -self.k_vz*self.v_ned[2])*self.m

        ax = a_2d_d[0]
        ay = a_2d_d[1]
        # Guidance attitude
        phi_d = -self.m/self.T_d*(ay*np.cos(self.att[2])-ax*np.sin(self.att[2]))
        the_d =  self.m/self.T_d*(ax*np.cos(self.att[2])+ay*np.sin(self.att[2]))

        # Control motors
        self.att_d = np.array([phi_d, the_d, self.yaw_d])
        if self.att_con == 0:
            self.control_att()
        elif self.att_con == 1:
            self.control_att_geometric()

        self.w_d = np.sqrt(self.Tlmn_to_w.dot(np.append(self.T_d, self.lmn_d)))

    def step_estimator_xi_g(self, dt):
        self.xi_g = self.xi_g + self.k_xi_g_v*self.v_ned[2]*dt \
        + self.k_xi_g_e_alt*self.e_alt*dt

    # Input: Desired 2D velocity, Desired altitude
    def set_v_2D_alt_lya(self, vxy_d, alt_d):
        e_alt  = self.xyz[2] - alt_d
        self.e_alt = e_alt
        vxy = self.v_ned[0:2]
        e_v = vxy - vxy_d
        self.e_v = e_v

        self.T_d = (-self.xi_g -self.k_alt*e_alt \
                -self.k_vz*self.v_ned[2])*self.m


        axy = self.xi_CD*la.norm(vxy)*vxy -self.k_vxy*e_v
        ax = axy[0]
        ay = axy[1]
        # Guidance attitude
        phi_d = -self.m/self.T_d*(ay*np.cos(self.att[2])-ax*np.sin(self.att[2]))
        the_d =  self.m/self.T_d*(ax*np.cos(self.att[2])+ay*np.sin(self.att[2]))

        # Control motors
        self.att_d = np.array([phi_d, the_d, self.yaw_d])
        if self.att_con == 0:
            self.control_att()
        elif self.att_con == 1:
            self.control_att_geometric()

        self.w_d = np.sqrt(self.Tlmn_to_w.dot(np.append(self.T_d, self.lmn_d)))

    def step_estimator_xi_CD(self, dt):
        self.xi_CD = self.xi_CD \
                - self.k_xi_CD_e_v*la.norm(self.v_ned[0:2])*(self.e_v.T).dot(self.v_ned[0:2])*dt

    ### Physics Simulation ###
    def step(self, dt):
        self.step_rotors(dt)
        self.step_6DoF(dt)
        self.step_estimator_xi_g(dt)
        self.step_estimator_xi_CD(dt)

    def step_rotors(self, dt): # Motors modelled as 1st order linear system
        # Check Saturation 
        for i in range (0, 4):
            if self.w_d[i] < 0:
                self.w_d[i] = 0
            elif self.w_d[i] > 500:
                self.w_d[i] = 500

        e_w = self.w - self.w_d
        w_dot = -self.kw*np.identity(4).dot(e_w)
        self.w = self.w + w_dot*dt

    def step_6DoF(self, dt):
        Rbn = self.Rot_bn() # Rotational matrix from Nav to Body
        g = np.array([0, 0, 9.81]) # Gravity vector
        p_dot = Rbn.dot(self.v_ned) # Velocity in body coordinates

        self.rotors_forces_moments() # Forces and moments by motors
        self.aero_forces_moments()   # Forces and moments by environment

        # Time derivatives (acc and vel) given by physics equations
        att_dot = (self.R_pqr()).dot(self.pqr)
        p_ddot = (self.Ft + self.Fa)/self.m + Rbn.dot(g) \
                - np.cross(self.pqr, p_dot)
        pqr_dot = self.Jinv.dot(self.Mt + self.Ma \
                - np.cross(self.pqr, self.J.dot(self.pqr)))

        # Propagation of positions/angles and velocities
        self.att = self.att + att_dot*dt
        #for i in range(0,3):
        #    self.att[i] = self.norm_ang(self.att[i])
        self.pqr = self.pqr + pqr_dot*dt

        # Touching the ground?
        if self.xyz[2] > 0:
            self.xyz[2] = 0
            if la.norm(self.v_ned) > 0.5:
                print self.tag, "crashed into the ground"
                self.crashed = 1
            self.v_ned[0:3] = 0
        else:
            self.v_ned = self.v_ned + Rbn.T.dot(p_ddot)*dt
            self.xyz = self.xyz + self.v_ned*dt

    # Forces and moments given by motors and environment
    def rotors_forces_moments(self):
        Tlmn = self.w_to_Tlmn.dot(np.array([self.w[0]**2, \
                self.w[1]**2, self.w[2]**2, self.w[3]**2]))

        self.Ft = np.array([0, 0, Tlmn[0]]) # Thrust
        self.Mt = Tlmn[1:4] # Moment

    def aero_forces_moments(self):
        Rbn = self.Rot_bn()
        p_dot = Rbn.dot(self.v_ned)

        Dl = -p_dot*la.norm(p_dot)*self.CDl # Linear drag
        Dr = -self.pqr*la.norm(self.pqr)*self.CDr # Angular drag
        
        self.Fa = Dl # Forces by the environment
        self.Ma = Dr # Moments by the environment

    ### Misc ###
    # Angles always between -pi and pi
    def norm_ang(self, x):
        if x > np.pi:
            x = x - 2*np.pi
        elif x <= -np.pi:
            x = x + 2*np.pi
        return x
    
    # Rotational matrix from Nav to Body
    def Rot_bn(self):
        phi = self.att[0]
        theta = self.att[1]
        psi = self.att[2]
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        cthe = np.cos(theta)
        sthe = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)

        Rx = np.array([[1,    0,      0], \
                       [0,  cphi,  sphi], \
                       [0, -sphi,  cphi]])

        Ry = np.array([[cthe,  0,  -sthe], \
                       [   0,  1,      0], \
                       [sthe,  0,   cthe]])

        Rz = np.array([[ cpsi,  spsi, 0], \
                       [-spsi,  cpsi, 0], \
                       [    0,    0, 1]])

        R = Rx.dot(Ry).dot(Rz)
        return R

    # Rotation matrix from Nav to given Body attitude
    def Rotd_bn(self, phi, theta, psi):
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        cthe = np.cos(theta)
        sthe = np.sin(theta)
        cpsi = np.cos(psi)
        spsi = np.sin(psi)

        Rx = np.array([[1,    0,      0], \
                       [0,  cphi,  sphi], \
                       [0, -sphi,  cphi]])

        Ry = np.array([[cthe,  0,  -sthe], \
                       [   0,  1,      0], \
                       [sthe,  0,   cthe]])

        Rz = np.array([[ cpsi,  spsi, 0], \
                       [-spsi,  cpsi, 0], \
                       [    0,    0, 1]])

        R = Rx.dot(Ry).dot(Rz)
        return R

    
    # Propagation matrix for computing the angular velocity of the attitude
    def R_pqr(self):
        phi = self.att[0]
        theta = self.att[1]
        tthe = np.tan(theta)
        cthe = np.cos(theta)
        cphi = np.cos(phi)
        sphi = np.sin(phi)

        R = np.array([[1, tthe*sphi, tthe*cphi], \
                      [0,      cphi,     -sphi], \
                      [0, sphi/cthe, cphi/cthe]])
        return R

    # Building a tensor from vector and viceversa
    def build_tensor_from_vector(self, a, b, c):
        T = np.array([[ 0, -c,  b],
                      [ c,  0, -a],
                      [-b,  a,  0]])
        return T

    def build_vector_from_tensor(self, T):
        v = np.array([T[2, 1], T[0, 2], T[1, 0]])
        return v
