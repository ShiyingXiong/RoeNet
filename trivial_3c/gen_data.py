import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import h5py
import os


# Define parameters for the simulation
igst = 10
grid_size = 200
xs = -0.5
xe = 0.5
lx = xe - xs
dx = lx / grid_size
x0 = torch.tensor(range(grid_size), dtype=torch.float32, requires_grad=True).unsqueeze(0).unsqueeze(0) * lx / grid_size + xs
m, n, l = x0.shape
x0f = torch.zeros(m, n, l + igst * 2)
x0f[:, :, 0:igst] = x0[:, :, l - igst:l] - lx
x0f[:, :, l + igst:l + igst * 2] = x0[:, :, 0:igst] + lx
x0f[:, :, igst:l + igst] = x0[:, :, 0:l]
tstart = 0
n_steps = 100
nconp = 3
L = torch.tensor([0.1,0.2,0.3,-0.5,-0.9,1,-0.1,0.4,0.8]).reshape([nconp,nconp])
R = torch.inverse(L)
Lambda = torch.tensor([1.,0.,0.,0.,1.,0,0.,0.,-1]).reshape([nconp,nconp])
A = R@Lambda@L
print(A)
Lam = torch.tensor([1,1,-1])
          
                
#Address boudary conditions
def cal_boundry(u, igst):
    m, n, l = u.shape
    u = torch.cat((u[:, :, igst:igst+1].expand(m,n,igst), u[:, :, igst: -igst], u[:, :, -igst-1:-igst].expand(m,n,igst)), dim=2)
    return u

# 4th order Runge Kutta scheme
def runge_kutta(z0, t1_t0, f, eps=0.001):
    n_steps = round(t1_t0 / eps)
    h = t1_t0 / n_steps
    z = z0
    for i_step in range(int(n_steps)):
        k1 = cal_boundry(z + h * f(z), igst)
        k2 = cal_boundry(0.75*z + 0.25*k1 + 0.25 * h * f(k1), igst)
        z = cal_boundry(z/3. + 2.*k2/3. + 2. * h * f(k2)/3., igst)
    return z

# Convert PyTorch tensor to numpy array
def to_np(x):
    return x.detach().cpu().numpy()

# Definition of a neural ODE model
class NeuralODE(nn.Module):
    def __init__(self, func, tol=1e-3):
        super(NeuralODE, self).__init__()
        self.func = func
        self.tol = tol

    def forward(self, z0, t1_t0):
        return runge_kutta(z0, t1_t0, self.func, self.tol)


#Analytic solution function
def any_solution(x0f,t):
    ul = [0.4,0.4,0.4]
    ur = [-0.4,-0.4,-0.4]
    
    m, n, l = x0f.shape
    data = torch.zeros(nconp,l)
    datau  = torch.zeros(nconp,l)
    for index in range(l):
        for i in range(nconp):
            if (x0f[0,0,index] <= Lam[i]*t):
                data[i,index] = ul[i]
            else:
                data[i,index] = ur[i]
        for i in range(nconp):
            for j in range(nconp):
                datau[i, index] = datau[i, index] +R[i,j] * data[j,index]
    return datau

#Construct RoeNet
class Roe(nn.Module):
    def __init__(self, dx):
        super(Roe, self).__init__()
        self.dx = dx

    def forward(self, u):
        ul = torch.cat((u[:, :, 0:1], u[:, :, :-1]), 2)
        ur = torch.cat((u[:, :, 1:], u[:, :, -1:]), 2)
        m, n, l = u.shape
        du = ur - u
        Rur = torch.zeros(m,n,l)
        for i in range(nconp):
            for j in range(nconp):
                for k in range(nconp):
                    Rur[:,k,:] = Rur[:,k,:]+(Lam[i]-torch.abs(Lam[i]))*L[i,j]*R[k,i]*du[:,j,:]
             
        du = u - ul
        Rul = torch.zeros(m,n,l)
        for i in range(nconp):
            for j in range(nconp):
                for k in range(nconp):
                    Rul[:,k,:] = Rul[:,k,:]+(Lam[i]+torch.abs(Lam[i]))*L[i,j]*R[k,i]*du[:,j,:]
             
        return -(Rur + Rul) / (2 * self.dx)

# Generate analytic solution
def gen_Any_data():
    DT = 0.02
    data_uC0 = any_solution(x0f,tstart).unsqueeze(0)
    with torch.no_grad():
        for i in range(1,n_steps+1):
            t = i*DT+tstart
            print(t)
            tp = any_solution(x0f,t).unsqueeze(0)
            data_uC0 = torch.cat((data_uC0,tp),0).float()
    torch.save(data_uC0, 'uAny.dat')

# Generate data for testing
def gen_test_data():
    data_uC0 = []
    data_DT = []
    DT = 0.02
    uC0 = any_solution(x0f,tstart).unsqueeze(0)
    ff = Roe(dx)
    ff.eval()
    with torch.no_grad():
        for i in range(n_steps+1):
            t = i*DT+tstart
            print(t)
            data_uC0.append(uC0)
            data_DT.append(t)
            uC0 = runge_kutta(uC0, DT, ff, 1e-3)
    data_uC0 = torch.cat(data_uC0).float()
    torch.save(data_uC0, 'uRoe.dat')

#Plotting the results
def plot_u(x0, ur):
    plt.clf()
    x = to_np(torch.squeeze(x0))
    ur = to_np(torch.squeeze(ur))
    plt.clf()
    plt.scatter(x, ur)
    plt.xlim(-0.5, 0.5)
    plt.ylim(-6, 6)
    plt.draw()
    plt.pause(0.1)
          
# Generate data for training    
def gen_data():
    data_uC0 = []
    data_uC1 = []
    data_DT = []
    n_steps = 1000
    DT = 0.02
    with torch.no_grad():
        for i in range(n_steps):
            print(i)
            t0 = np.random.uniform(0.04, 0.08)
            uC0 = any_solution(x0f,t0).unsqueeze(0)
            #DT = np.random.uniform(0.01, 0.05)
            uC1 = any_solution(x0f,t0+DT).unsqueeze(0)
            data_uC0.append(uC0)
            data_uC1.append(uC1)
            if (i<100):
                plot_u(x0f, uC0[:, 0, :])
                plot_u(x0f, uC1[:, 0, :])
            data_DT.append(float(DT))
    data_uC0 = torch.cat(data_uC0).float()
    data_uC1 = torch.cat(data_uC1).float()
    data_DT = torch.tensor(data_DT).float()
    data_root = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    hf = h5py.File(os.path.join(data_root, "data.h5"), "w")
    hf.create_dataset('uC0', data=data_uC0)
    hf.create_dataset('uC1', data=data_uC1)
    hf.create_dataset('DT', data=data_DT)
    hf.close()

# Execute the data generation functions
gen_data()
gen_test_data()
gen_Any_data()
