import numpy as np
from scipy import linalg as la


# The symbols are taken from the PhD Thesis "Distributed formation control for autonomous robots" University of Groningen.

class formation_distance:
    def __init__(self, m, l, d, mu, tilde_mu, B, c_shape, c_vel):
        self.m = m # Dimension {2,3}
        self.l = l # Order of the Lyapunov function (1,2,...)
        self.d = d # Set of desired distances (linked to the set of edges)
        self.mu = mu # Set of mismatches
        self.tilde_mu = tilde_mu # Set of mismatches
        self.B = B # Incidence matrix
        self.agents, self.edges = self.B.shape
        self.S1 = self.make_S1() # Aux matrix
        self.S2 = self.make_S2() # Aux matrix
        self.Av = self.make_Av() # Matrix A for eventual velocities
        self.Aa = self.Av.dot(self.B.T).dot(self.Av) # Matrix A for eventual acceleration

        # Kronecker products
        self.Bb = la.kron(self.B, np.eye(self.m))
        self.S1b = la.kron(self.S1, np.eye(self.m))
        self.S2b = la.kron(self.S2, np.eye(self.m))
        self.Avb = la.kron(self.Av, np.eye(self.m))
        self.Aab = la.kron(self.Aa, np.eye(self.m))

        # Distance error vector
        self.Ed = np.zeros(self.edges)

        # Velocity error vector
        self.Ev = np.zeros(self.edges*self.m)

        # Gain controllers
        self.c_shape = c_shape
        self.c_vel = c_vel

    # Desired acceleration given \ddot p = u
    def u_acc(self, X, V):
        Z = self.Bb.T.dot(X)
        Dz = self.make_Dz(Z)
        Dzt = self.make_Dzt(Z)
        self.Ed = self.make_E(Z)
        U = -self.c_shape*self.Bb.dot(Dz).dot(Dzt).dot(self.Ed) + \
                self.c_vel*self.Avb.dot(Z) + self.Aab.dot(Z) \
                - self.c_vel*V
        return U

    # Desired velocities given \dot p = u
    def u_vel(self, X):
        Z = self.Bb.T.dot(X)
        Dz = self.make_Dz(Z)
        Dzt = self.make_Dzt(Z)
        self.Ed = self.make_E(Z)
        U = -self.c_shape*self.Bb.dot(Dz).dot(Dzt).dot(self.Ed) \
                + self.Avb.dot(Z)
        return U

    # Construct S1 from B
    def make_S1(self):
        S1 = np.zeros_like(self.B)

        for i in range(0, self.agents):
            for j in range(0, self.edges):
                if self.B[i,j] == 1:
                    S1[i,j] = 1

        return S1

    # Construct S2 from B
    def make_S2(self):
        S2 = np.zeros_like(-self.B)

        for i in range(0, self.agents):
            for j in range(0, self.edges):
                if self.B[i,j] == 1:
                    S2[i,j] = 1
        return S2

    # Incidence matrix Bd given the first agent is a leader
    def make_Bd(self):
        Bd = np.copy(self.B)

        for i in range(0, self.edges):
            Bd[0, i] = 0

        return Bd

    # Diagonal matrix spliting the z elements of Z
    def make_Dz(self, Z):
        Dz = np.zeros((Z.size, self.edges))
    
        j = 0

        for i in range(0, self.edges):
            Dz[j:j+self.m, i] =  Z[j:j+self.m]
            j+=self.m

        return Dz

    # Diagonal matrix spliting the z/||z|| elements of Z
    def make_Dzt(self, Z):
        if self.l == 2:
            return np.eye(self.edges)

        Zt = np.zeros(self.edges)
        for i in range(0, self.edges):
            Zt[i] = (la.norm(Z[(i*self.m):(i*self.m+self.m)]))**(self.l-2)

        return np.diag(Zt)

    # Construct distance error vector
    def make_E(self, Z):
        E = np.zeros(self.edges)
        for i in range(0, self.edges):
            E[i] = (la.norm(Z[(i*self.m):(i*self.m+self.m)]))**self.l \
                    - self.d[i]**self.l

        return E

    # Construct A matric for desired velocities
    def make_Av(self):
        A = np.zeros(self.B.shape)
        for i in range(0, self.agents):
            for j in range(0, self.edges):
                if self.B[i,j] == 1:
                    A[i,j] = self.mu[j]
                elif self.B[i,j] == -1:
                    A[i,j] = self.tilde_mu[j]

        return A


