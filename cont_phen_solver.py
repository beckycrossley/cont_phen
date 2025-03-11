#solver for continuous phenotype model

import numpy as np
from scipy.integrate import ode

#Here we adapt the p10 solver from 'add_in_int_ecm.ipynb'. 
class solver():
    #u1 is cells - 2d
    #u2 is matrix - 1d
    def __init__(self,Nx, Ny, dt, tmax, Dx=1, Dy=1, r=1, umax=1, deg=1, v=1, xmin=0, xmax=1, ymin=0, ymax=1):
        self.Nx = Nx
        self.Ny = Ny
        self.dt = dt
        self.tmax = tmax
        self.Dx = Dx
        self.Dy = Dy
        self.r = r
        self.umax = umax
        self.deg = deg        
        self.v = v
        

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        
        self.dx = (self.xmax - self.xmin) / self.Nx
        self.dy = (self.ymax - self.ymin) / self.Ny
        self.spx = np.linspace(self.xmin, self.xmax, self.Nx + 1)
        self.spy = np.linspace(self.ymin, self.ymax, self.Ny + 1)
        
        # self.dx=1/self.Nx; 
        # self.dy=1/self.Ny;
        # self.spx=np.linspace(0,self.Nx,self.Nx+1)*self.dx;
        # self.spy=np.linspace(0,self.Ny,self.Ny+1)*self.dy;
        self.ymean=self.meanval(self.spy);
        
        self.tim=np.arange(0,self.tmax+self.dt, self.dt); #was previously tmax+1 but unsure why - think on it now whilst do
    
    def set_ICs(self, init_u1, init_u2):
        #initialise density
        self.u1 = [init_u1]
        self.u2 = [init_u2]
        self.u_ex = [np.concatenate([init_u1.ravel(),init_u2])] # flatten the 2d array, followed by the 1d array (matrix)
        
    def meanval(self,z): #computes the mean value of the grid as half of the sum of the edges
        y=0.5*(z[:-1]+z[1:]);
        return y
       
    def splitter(self,u_expanded): #split u into arrays for each x component 
        u1 = u_expanded[:(self.Nx+1)*self.Ny].reshape((self.Nx+1,self.Ny))
        u2 = u_expanded[(self.Nx+1)*self.Ny:]
        return u1, u2
    
    def fun_local_mass(self, Usol): #computes the local mass of the solution
        return np.sum(Usol,axis=1) #self.hz*np.sum(Usol,axis=1) #computes actual mass not just total denisty which is what it currently does

    def set_diffusion_fct_y(self, diff_fct_y):
        self.diff_fct = diff_fct_y[0]
        
        self.diff_fct_names=[]
        for i in diff_fct_y:
            self.diff_fct_names.append(i.__name__)
            
    def set_advection_fct_y(self, adv_fct_y):
        self.adv_fct = adv_fct_y[0]
        
        self.adv_fct_names=[]
        for i in adv_fct_y:
            self.adv_fct_names.append(i.__name__)
    
    def disc_1D_inhomogeneous_x(self, f, D):
        dx = self.dx
        lap = np.zeros(f.shape)
        # Debugging print statements, commented out for clarity
        # print(D.shape, f.shape, lap.shape)
        # print(D[1:].shape, D[:-1].shape, D[1:][:, None].shape, f[2:, :].shape, f[1:-1, :].shape, f[:-2, :].shape,)

        # Adjusted for a 1D D along y
        lap[1:-1, :] = (
            ((D[2:] + D[1:-1])[:, None] * (f[2:, :] - f[1:-1, :]) -
            (D[1:-1] + D[:-2])[:, None] * (f[1:-1, :] - f[:-2, :])) / (2*dx**2)
        )

        # Boundary conditions
        lap[0, :] = (D[1] + D[0]) * (f[1, :] - f[0, :]) / (2*dx**2)
        lap[-1, :] = (D[-1] + D[-2]) * (f[-2, :] - f[-1, :]) / (2*dx**2)

        return lap

    def laplacian_y(self, f): # laplacian of density in x direction only
        dy = self.dy
        lap = np.zeros(f.shape) # zero array of same shape as f
        lap[:,1:-1] = (f[:,2:] - 2*f[:,1:-1] + f[:,:-2])/dy**2 # laplacian of f in inner points
        lap[:,0] = (f[:,1] - f[:,0])/dy**2
        lap[:,-1] = (f[:,-2] - f[:,-1])/dy**2
        return lap
  
    def laplacian_x(self, f): # laplacian of density in y direction only
        dx = self.dx
        lap = np.zeros(f.shape) # zero array of same shape as f
        lap[1:-1, :] = (f[2:, :] - 2*f[1:-1, :] + f[:-2,:])/dx**2 # laplacian of f in inner points
        lap[0, :] = (f[1, :] - f[0, :])/dx**2
        lap[-1, :] = (f[-2, :] - f[-1, :])/dx**2
        return lap
    
    def limiter(self, r):
        # Vectorized limiters
        delta = 2
        K = 1. / 3 + 2. / 3. * r
        y = np.maximum(0, np.minimum(np.minimum(2 * r, delta), K))
        y[np.logical_or(np.isinf(r), np.isnan(r))] = 0
        return y
    
    def prol(self,z,rho,m): #calculating proliferation rate
        """
        Parameters
        ----------
        z : vector
            phenotypic coordinate.
        rho : float
            local total cell number.
        m : float
            ECM density.
        par : dictionary
            dictionary containing the list of parameter involved in the model.
        Returns
        -------
        y : vector
        proliferation rate as a function of the phenotypic variable

        """
        
        vf_coefficient = (1 - (rho+m)/self.umax)

        y = self.r*vf_coefficient
        return y
    
    def dyz(self, p1, p2):
        np.seterr(divide='ignore', invalid='ignore')

        # Vectorized computation of velocity, second order difference, and limiters
        vel = self.adv_fct(p1, p2, self.spy)

        u = p1        
        r = np.diff(u[:, 1:]) / np.diff(u[:, :-1])
        rp = np.insert(r, 0, 0, axis=1)
        rm = np.append(1 / r, np.zeros((r.shape[0], 1)), axis=1)
        
        phip = self.limiter(rp)
        phim = self.limiter(rm)
        
        fluxp = u[:, :-1] + 0.5 * phip * np.diff(u, axis=1)
        fluxm = u[:, 1:] + 0.5 * phim * np.diff(u[:, ::-1], axis=1)[:, ::-1]
        #print('vel:')
        #print(vel.shape)
        T = np.maximum(0, vel[:, 1:-1]) * fluxp + np.minimum(0, vel[:, 1:-1]) * fluxm
        # #print('check')
        # transport = np.diff(T, axis=1, prepend=T[:, 0:1], append=-T[:, -1:]) / self.dy
        
        transport = np.insert(np.diff(T, axis=1), 0, T[:, 0], axis=1)
        transport = np.append(transport, -T[:, -1:], axis=1) / self.dy

        return transport
    
    def diff_y(self, phen, dens, ecm):
        return self.diff_fct(phen, dens, ecm)
    
    def adv_y(self, phen, dens, ecm):
        return self.adv_fct(phen, dens, ecm)
    
    def dpdt(self, t, p_expanded):
        p1, p2 = self.splitter(p_expanded)
        
         # Sum p1 across the y direction
        #print(p1.shape)
        p1_summed_over_y = self.fun_local_mass(p1)
        #print(p1_summed_over_y.shape, p2.shape)
        #np.sum(p1, axis=0)  # This creates a 1D array
        vf_coefficient = (1 - p1_summed_over_y/self.umax - p2/self.umax)
        vf_coefficient2 = vf_coefficient[:, np.newaxis] # This creates a 2D array
        
        # Existing diffusion terms
        h1 = self.disc_1D_inhomogeneous_x(p1, vf_coefficient) 
        diff_y_result = self.diff_y(self.ymean, p1_summed_over_y, p2)
        #print("Shape of diff_y_result:", diff_y_result.shape)
        h2 = self.laplacian_y(self.diff_y(self.ymean, p1_summed_over_y, p2)*p1) #this is the line we change for nonlinear diffusion!!!!!!!!!!!!!!!
        
        zcomp = self.dyz(p1, p2)
        
        # Combine diffusion, advection, and reaction terms
        diff_1 = self.Dx*h1*(1-self.ymean) + self.Dy*h2 +self.v*zcomp + self.r*p1*vf_coefficient2*self.ymean #(1-p1/self.umax) # self.v*h4
        
        # Reaction term for ECM degradation by cells
        deg_int = self.fun_local_mass(p1*(1-self.ymean))
        #diff_2 = -self.deg*p2*p1_summed_over_y          #/self.umax
        diff_2 = -self.deg*p2*deg_int
        
        return np.concatenate([diff_1.ravel(), diff_2])
    
    def solve(self):
        print('Maximum time = %.2f' % self.tmax)
        
        solODE = ode(self.dpdt).set_integrator('dopri5')
        solODE.set_initial_value(self.u_ex[0],0)
        
        t = 0
        k = 0
        while t < self.tmax:
            print('t = %.2f'% t,end = '\r')
            t += self.dt
            k += 1
            
            u_ex_next = solODE.integrate(t)
            p1, p2 = self.splitter(u_ex_next)
            self.u1.append(p1)
            self.u2.append(p2)
        t_end = t + self.dt
        print('Finished: t = %.2f'% t_end)
        return self.u1, self.u2    