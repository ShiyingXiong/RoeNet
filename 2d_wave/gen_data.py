import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import torch.nn.functional as F
import torch.nn as nn
import random
import h5py
import os

igst = 10
grid_size = 100
xs = -0.6
xe = 0.6
lx = xe - xs
dx = lx / grid_size
x0 = torch.tensor(range(grid_size+2*igst), dtype=torch.float32, requires_grad=True).unsqueeze(0).unsqueeze(0) * lx / (grid_size+2*igst) + xs
y0 = torch.tensor(range(grid_size+2*igst), dtype=torch.float32, requires_grad=True).unsqueeze(0).unsqueeze(0) * lx / (grid_size+2*igst) + xs
m, n, l = x0.shape
x0f = x0
y0f = y0
tstart = 0.0
n_steps = 101
nconp = 1
#L = torch.randn([nconp,nconp])
L = torch.tensor([1.]).reshape([nconp,nconp])
R = torch.inverse(L)
#Lam = torch.randn(nconp)
Lam = torch.tensor([1.])
def cal_boundry(u, igst):
    m, n, l = u.shape
    u = torch.cat((u[:, :, -2*igst:-igst], u[:, :, igst: -igst], u[:, :, igst:2*igst]), dim=2)
    return u


def runge_kutta(z0, t1_t0, f, eps=0.001):
    n_steps = round(t1_t0 / eps)
    h = t1_t0 / n_steps
    z = z0
    for i_step in range(int(n_steps)):
        k1 = cal_boundry(z + h * f(z), igst)
        k2 = cal_boundry(0.75*z + 0.25*k1 + 0.25 * h * f(k1), igst)
        z = cal_boundry(z/3. + 2.*k2/3. + 2. * h * f(k2)/3., igst)
    return z


def to_np(x):
    return x.detach().cpu().numpy()


class NeuralODE(nn.Module):
    def __init__(self, func, tol=1e-3):
        super(NeuralODE, self).__init__()
        self.func = func
        self.tol = tol

    def forward(self, z0, t1_t0):
        return runge_kutta(z0, t1_t0, self.func, self.tol)

    
class CentDif(nn.Module):
    def __init__(self, dx, igst):
        super(CentDif, self).__init__()
        self.dx = dx
        self.igst = igst
        # self.nu = 0.001

    def forward(self, u):
        m, n, l = u.shape
        #u = torch.squeeze(u)
        du = torch.zeros(m,n,l)
        # ddu = torch.zeros(l)
        for i in range(igst, l - igst):
            du[:,:,i] = (-u[:,:,i - 3] + 9. * u[:,:,i - 2] - 45. * u[:,:,i - 1] + 45. * u[:,:,i + 1] - 9. * u[:,:,i + 2] + u[:,:,i + 3]) / (60. * self.dx)
        adu = torch.zeros(m,n,l)
        adu[:,0,:] = u[:,1,:]*du[:,0,:]+u[:,0,:]*du[:,1,:]
        adu[:,1,:] = 2*du[:,0,:]+u[:,1,:]*du[:,1,:]
        return -adu

def any_solution(x0f,y0f,t):
    m, n, h = x0f.shape
    m, n, w = x0f.shape
    u = torch.zeros(1,h,w)
    for idx1 in range(h):
        for idx2 in range(w):
            tp = (x0f[0,0,idx1] + y0f[0,0,idx2] + t) * 2 * 3.1415927
            u[0,idx1, idx2] = torch.sin(tp)
    return u

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


def plot_u(x0, y0, ur, i):
    fig = plt.figure()
    ax = Axes3D(fig)
    x = to_np(torch.squeeze(x0))
    y = to_np(torch.squeeze(y0))
    ur = to_np(ur)
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y,ur)
    plt.savefig(i+'.jpg')
    
def gen_data():
    data_uC0 = []
    data_uC1 = []
    data_DT = []
    n_steps = 200
    DT = 0.1
    with torch.no_grad():
        for i in range(n_steps):
            # print(i)
            t0 = np.random.uniform(0.0, 0.1)
            uC0 = any_solution(x0f,y0f,t0).unsqueeze(0)
            #DT = np.random.uniform(0.01, 0.05)
            uC1 = any_solution(x0f,y0f,t0+DT).unsqueeze(0)#+torch.randn(uC0.shape)*0.1
            data_uC0.append(uC0)
            data_uC1.append(uC1)
            if (i<5):
                plot_u(x0f, y0f, uC0[0, 0, :, :], str(i)+'c0')
                plot_u(x0f, y0f, uC1[0, 0, :, :], str(i)+'c1')
            if (i % 10 == 0):
                print (i)
            data_DT.append(float(DT))
    data_uC0 = torch.cat(data_uC0).float()
    data_uC1 = torch.cat(data_uC1).float()
    data_DT = torch.tensor(data_DT).float()
    data_root = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    hf = h5py.File(os.path.join(data_root, "data.h5"), "w")
    print(data_uC0.shape)
    hf.create_dataset('uC0', data=data_uC0)
    hf.create_dataset('uC1', data=data_uC1)
    hf.create_dataset('DT', data=data_DT)
    hf.close()

if __name__ == '__main__':
    gen_data()
    gen_test_data()
    gen_Any_data()