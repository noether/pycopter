from scipy import linalg as la
import matplotlib.pyplot as pl
import numpy as np

import quadrotor as quad
import quadlog
import animation as ani

# Quadrotor
m = 0.65 # Kg
l = 0.23 # m
Jxx = 7.5e-3 # Kg/m^2
Jyy = Jxx
Jzz = 1.3e-2
Jxy = 0
Jxz = 0
Jyz = 0
J = np.array([[Jxx, Jxy, Jxz], \
              [Jxy, Jyy, Jyz], \
              [Jxz, Jyz, Jzz]])
CDl = 9e-3
CDr = 9e-4
kt = 3.13e-5  # Ns^2
km = 7.5e-7   # Ns^2
kw = 1/0.18   # rad/s

# Initial conditions
att_0 = np.array([0.0, 0.0, 0.0])
pqr_0 = np.array([0.0, 0.0, 0.0])
xyz_0 = np.array([0.0, 0.0, 0.0])
v_ned_0 = np.array([0.0, 0.0, 0.0])
w_0 = np.array([0.0, 0.0, 0.0, 0.0])

# Setting quads
q1 = quad.quadrotor(1, m, l, J, CDl, CDr, kt, km, kw, \
        att_0, pqr_0, xyz_0, v_ned_0, w_0)

# Simulation parameters
tf = 250
dt = 1e-2
time = np.linspace(0, tf, tf/dt)
it = 0
frames = 100

# Data log
q1_log = quadlog.quadlog(time)

# Plots
quadcolor = ['k']
pl.close("all")
pl.ion()
fig = pl.figure(0)
axis3d = fig.add_subplot(111, projection='3d')

# Desired position and heading
v_2D_d = np.array([2, -0.5])
alt_d = -8
q1.yaw_d = np.pi/3

for t in time:

    # Simulation
    q1.set_v_2D_alt_lya(v_2D_d, alt_d)
    q1.step(dt)

    # Animation
    if it%frames == 0:

        pl.figure(0)
        axis3d.cla()
        ani.draw3d(axis3d, q1.xyz, q1.Rot_bn(), quadcolor[0])
        axis3d.set_xlim(-10, 10)
        axis3d.set_ylim(-10, 10)
        axis3d.set_zlim(0, 15)
        axis3d.set_xlabel('South [m]')
        axis3d.set_ylabel('East [m]')
        axis3d.set_zlabel('Up [m]')
        axis3d.set_title("Time %.3f s" %t)
        pl.pause(0.001)
        pl.draw()
        

    # Log
    q1_log.xyz_h[it, :] = q1.xyz
    q1_log.att_h[it, :] = q1.att
    q1_log.w_h[it, :] = q1.w
    q1_log.v_ned_h[it, :] = q1.v_ned
    q1_log.xi_g_h[it] = q1.xi_g
    q1_log.xi_CD_h[it] = q1.xi_CD

    it+=1
    
    # Stop if crash
    if (q1.crashed == 1):
        break

pl.figure(1)
pl.plot(time, q1_log.w_h[:, 0], label="w_1")
pl.plot(time, q1_log.w_h[:, 1], label="w_2")
pl.plot(time, q1_log.w_h[:, 2], label="w_3")
pl.plot(time, q1_log.w_h[:, 3], label="w_4")
pl.xlabel("Time [s]")
pl.ylabel("Motor angular velocity [rad/s]")
pl.grid()
pl.legend()

pl.figure(2)
pl.plot(time, q1_log.att_h[:, 0], label="roll")
pl.plot(time, q1_log.att_h[:, 1], label="pitch")
pl.plot(time, q1_log.att_h[:, 2], label="yaw")
pl.xlabel("Time [s]")
pl.ylabel("Attitude angle [rad]")
pl.grid()
pl.legend()

pl.figure(3)
pl.plot(time, -q1_log.xyz_h[:, 2], label="UP")
pl.plot(time, q1_log.xyz_h[:, 0], label="X")
pl.plot(time, q1_log.xyz_h[:, 1], label="Y")
pl.xlabel("Time [s]")
pl.ylabel("Position [m]")
pl.grid()
pl.legend()

pl.figure(4)
pl.plot(time, -q1_log.v_ned_h[:, 2], label="-V_z")
pl.plot(time, q1_log.v_ned_h[:, 0], label="V_x")
pl.plot(time, q1_log.v_ned_h[:, 1], label="V_y")
pl.xlabel("Time [s]")
pl.ylabel("Velocity [m/s]")
pl.grid()
pl.legend()

pl.figure(5)
pl.plot(time, q1_log.xi_g_h, label="${\\xi}_g$")
pl.plot(time, q1_log.xi_CD_h, label="${\\xi}_{CD}$")
pl.xlabel("Time [s]")
pl.ylabel("Estimators value")
pl.grid()
pl.legend()

pl.pause(0)
