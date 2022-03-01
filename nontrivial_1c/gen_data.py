import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import random
import h5py
import os

igst = 10
grid_size = 100
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
tstart = 0.0
n_steps = 80
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
    if (n_steps<1):
        n_steps = 1
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
    def __init__(self,dx):
        super(CentDif, self).__init__()
        self.dx = dx
    def forward(self,um):
        ul1 = torch.cat((um[:, :, -1:], um[:, :, :-1]), 2)
        ur1 = torch.cat((um[:, :, 1:], um[:, :, :1]), 2)
        ul2 = torch.cat((um[:, :, -2:], um[:, :, :-2]), 2)
        ur2 = torch.cat((um[:, :, 2:], um[:, :, :2]), 2)
        ul3 = torch.cat((um[:, :, -3:], um[:, :, :-3]), 2)
        ur3 = torch.cat((um[:, :, 3:], um[:, :, :3]), 2)
        du = (-ul3+9.*ul2-45.*ul1+45.*ur1-9.*ur2+ur3)/(60.*self.dx)
        return -um*du
    
def any_solution(x0f,t):
    u = 0.5 + torch.sin(x0f*2.*math.pi)
    dx = x0f[0,0,1]-x0f[0,0,0]
    ff = CentDif(dx)
    ff.eval()
    u = runge_kutta(u, t, ff, 0.001)
    return u

class Roe(nn.Module):
    def __init__(self, dx):
        super(Roe, self).__init__()
        self.dx = dx

    def forward(self, um):
        ul = torch.cat((um[:, :, -1:], um[:, :, :-1]), 2)
        ur = torch.cat((um[:, :, 1:], um[:, :, :1]), 2)
        Rur = (um+ur-torch.abs(um+ur))*(ur-um)
        Rul = (um+ul+torch.abs(um+ul))*(um-ul)
        return -(Rur + Rul) / (4. * self.dx)

def gen_Any_data():
    DT = 0.01
    data_uC0 = any_solution(x0f,tstart)
    with torch.no_grad():
        for i in range(1,n_steps+1):
            t = i*DT+tstart
            print(t)
            tp = any_solution(x0f,t)
            data_uC0 = torch.cat((data_uC0,tp),0).float()
    torch.save(data_uC0, 'uAny.dat')
    
def gen_test_data():
    data_uC0 = []
    data_DT = []
    DT = 0.01
    uC0 = any_solution(x0f,tstart)
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
    
def plot_u(x0, ur):
    plt.clf()
    x = to_np(torch.squeeze(x0))
    ur = to_np(torch.squeeze(ur))
    plt.clf()
    plt.scatter(x, ur)
    plt.xlim(-0.5, 0.5)
    plt.ylim(-1., 2.)
    plt.draw()
    plt.pause(0.01)
    
def gen_data():
    data_uC0 = []
    data_uC1 = []
    data_DT = []
    n_steps = 100
    DT = 0.001
    with torch.no_grad():
        for i in range(n_steps):
            print(i)
            t0 = np.random.uniform(0.0, 0.001)
            uC0 = any_solution(x0f,t0)
            #DT = np.random.uniform(0.01, 0.05)
            uC1 = any_solution(x0f,t0+DT)
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


gen_data()
gen_test_data()
#gen_Any_data()