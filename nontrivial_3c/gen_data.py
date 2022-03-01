import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import h5py
import os


def cal_boundry(u, igst):
    m, n, l = u.shape
    u = torch.cat((u[:, :, igst:igst + 1].expand(m, n, igst), u[:, :, igst: -igst], u[:, :, -igst - 1:-igst].expand(m, n, igst)), dim=2)
    return u


def runge_kutta(z0, t1_t0, f, eps=1e-3):
    n_steps = round(t1_t0 / eps)
    h = t1_t0 / n_steps
    z = z0
    for i_step in range(int(n_steps)):
        k1 = cal_boundry(z + h * f(z), igst)
        k2 = cal_boundry(0.75 * z + 0.25 * k1 + 0.25 * h * f(k1), igst)
        z = cal_boundry(z / 3. + 2. * k2 / 3. + 2. * h * f(k2) / 3., igst)
    return z


def to_np(x):
    return x.detach().cpu().numpy()



def any_solution(x0f, t):
    x0 = 0
    rho_l = 1
    P_l = 1
    u_l = 0

    rho_r = 0.125
    P_r = 0.1
    u_r = 0

    gamma = 1.4
    mu = np.sqrt((gamma - 1) / (gamma + 1))

    # speed of sound
    c_l = np.power((gamma * P_l / rho_l), 0.5)
    # c_r = np.power( (gamma*P_r/rho_r),0.5)

    P_post = 0.319000535309729
    v_post = 2 * (np.sqrt(gamma) / (gamma - 1)) * (1 - np.power(P_post, (gamma - 1) / (2 * gamma)))
    rho_post = rho_r * (((P_post / P_r) + np.power(mu, 2.)) / (1 + mu * mu * (P_post / P_r)))
    v_shock = v_post * ((rho_post / rho_r) / ((rho_post / rho_r) - 1))
    rho_middle = (rho_l) * np.power((P_post / P_l), 1 / gamma)
    m, n, l = x0f.shape

    # Key Positions
    x1 = x0 - c_l * t
    x3 = x0 + v_post * t
    x4 = x0 + v_shock * t
    # determining x2
    c_2 = c_l - ((gamma - 1) / 2) * v_post
    x2 = x0 + (v_post - c_2) * t

    # start setting values
    data = torch.zeros(4, l)
    for index in range(l):
        if (x0f[0, 0, index] < x1):
            # Solution b4 x1
            # data.rho(index) = rho_l
            # data.P(index) = P_l
            # data.u(index) = u_l
            data[0, index] = rho_l
            data[1, index] = P_l
            data[2, index] = u_l
        elif (x1 <= x0f[0, 0, index] and x0f[0, 0, index] <= x2):
            # Solution b/w x1 and x2
            c = mu * mu * ((x0 - x0f[0, 0, index]) / t) + (1 - mu * mu) * c_l
            # data.rho(index) = rho_l*power((c/c_l),2/(gamma - 1))
            # data.P(index) = P_l*power((data.rho(index)/rho_l),gamma)
            # data.u(index) = (1 - mu*mu)*( (-(x0-data.x(index))/t) + c_l)
            data[0, index] = rho_l * np.power((c.detach().numpy() / c_l), 2 / (gamma - 1))
            data[1, index] = P_l * np.power((data[0, index].detach().numpy() / rho_l), gamma)
            data[2, index] = (1 - mu * mu) * ((-(x0 - x0f[0, 0, index].detach().numpy()) / t) + c_l)
        elif (x2 <= x0f[0, 0, index] and x0f[0, 0, index] <= x3):
            # Solution b/w x2 and x3
            # data.rho(index) = rho_middle
            # data.P(index) = P_post
            # data.u(index) = v_post
            data[0, index] = rho_middle
            data[1, index] = P_post
            data[2, index] = v_post
        elif (x3 <= x0f[0, 0, index] and x0f[0, 0, index] <= x4):
            # Solution b/w x3 and x4
            # data.rho(index) = rho_post
            # data.P(index) = P_post
            # data.u(index) = v_post
            data[0, index] = rho_post
            data[1, index] = P_post
            data[2, index] = v_post
        elif (x4 < x0f[0, 0, index]):
            # Solution after x4
            # data.rho(index) = rho_r
            # data.P(index) = P_r
            # data.u(index) = u_r
            data[0, index] = rho_r
            data[1, index] = P_r
            data[2, index] = u_r
        data[3, index] = data[1, index] / ((gamma - 1) * data[0, index])
    return data


class Roe(nn.Module):
    def __init__(self, dx):
        super(Roe, self).__init__()
        self.dx = dx

    def forward(self, u):
        ul = torch.cat((u[:, :, 0:1], u[:, :, :-1]), 2)
        ur = torch.cat((u[:, :, 1:], u[:, :, -1:]), 2)
        m, n, l = u.shape
        gamma = 1.4
        pm = (gamma - 1.) * (u[:, 2:3, :] - 0.5 * u[:, 1:2, :] * u[:, 1:2, :] / u[:, 0:1, :])
        pr = (gamma - 1.) * (ur[:, 2:3, :] - 0.5 * ur[:, 1:2, :] * ur[:, 1:2, :] / ur[:, 0:1, :])
        pl = (gamma - 1.) * (ul[:, 2:3, :] - 0.5 * ul[:, 1:2, :] * ul[:, 1:2, :] / ul[:, 0:1, :])
        Hm = (u[:, 2:3, :] + pm) / u[:, 0:1, :]
        Hr = (ur[:, 2:3, :] + pr) / ur[:, 0:1, :]
        Hl = (ul[:, 2:3, :] + pl) / ul[:, 0:1, :]
        sqrtrhor = torch.sqrt(ur[:, 0:1, :])
        sqrtrhom = torch.sqrt(u[:, 0:1, :])
        sqrtrhol = torch.sqrt(ul[:, 0:1, :])
        sqrtRr = (sqrtrhor + sqrtrhom)
        sqrtRl = (sqrtrhom + sqrtrhol)

        u_r = (ur[:, 1:2, :] / sqrtrhor + u[:, 1:2, :] / sqrtrhom) / sqrtRr
        u_l = (ul[:, 1:2, :] / sqrtrhol + u[:, 1:2, :] / sqrtrhom) / sqrtRl

        H_r = (sqrtrhor * Hr + sqrtrhom * Hm) / sqrtRr
        H_l = (sqrtrhol * Hl + sqrtrhom * Hm) / sqrtRl

        c_r = torch.sqrt((gamma - 1.) * (H_r - 0.5 * u_r * u_r))
        c_l = torch.sqrt((gamma - 1.) * (H_l - 0.5 * u_l * u_l))

        dur = ur - u
        c1 = 0.5 * (gamma - 1.) / (c_r * c_r)
        c2 = 0.5 * u_r * u_r
        c3 = c_r / (gamma - 1.)
        Lur = torch.zeros(m, n, l)
        l1 = u_r - c_r
        l2 = u_r
        l3 = u_r + c_r
        Lur[:, 0:1, :] = (l1 - torch.abs(l1)) * c1 * ((c2 + u_r * c3) * dur[:, 0:1, :] - (u_r + c3) * dur[:, 1:2, :] + dur[:, 2:3, :])
        Lur[:, 1:2, :] = (l2 - torch.abs(l2)) * 2 * c1 * ((-c2 + c_r * c3) * dur[:, 0:1, :] + u_r * dur[:, 1:2, :] - dur[:, 2:3, :])
        Lur[:, 2:3, :] = (l3 - torch.abs(l3)) * c1 * ((c2 - u_r * c3) * dur[:, 0:1, :] - (u_r - c3) * dur[:, 1:2, :] + dur[:, 2:3, :])
        Rur = torch.zeros(m, n, l)
        Rur[:, 0:1, :] = Lur[:, 0:1, :] + Lur[:, 1:2, :] + Lur[:, 2:3, :]
        Rur[:, 1:2, :] = l1 * Lur[:, 0:1, :] + u_r * Lur[:, 1:2, :] + l3 * Lur[:, 2:3, :]
        Rur[:, 2:3, :] = (H_r - u_r * c_r) * Lur[:, 0:1, :] + c2 * Lur[:, 1:2, :] + (H_r + u_r * c_r) * Lur[:, 2:3, :]

        dul = u - ul
        c1 = 0.5 * (gamma - 1.) / (c_l * c_l)
        c2 = 0.5 * u_l * u_l
        c3 = c_l / (gamma - 1.)
        Lul = torch.zeros(m, n, l)
        l1 = u_l - c_l
        l2 = u_l
        l3 = u_l + c_l
        Lul[:, 0:1, :] = (l1 + torch.abs(l1)) * c1 * ((c2 + u_l * c3) * dul[:, 0:1, :] - (u_l + c3) * dul[:, 1:2, :] + dul[:, 2:3, :])
        Lul[:, 1:2, :] = (l2 + torch.abs(l2)) * 2 * c1 * ((-c2 + c_l * c3) * dul[:, 0:1, :] + u_l * dul[:, 1:2, :] - dul[:, 2:3, :])
        Lul[:, 2:3, :] = (l3 + torch.abs(l3)) * c1 * ((c2 - u_l * c3) * dul[:, 0:1, :] - (u_l - c3) * dul[:, 1:2, :] + dul[:, 2:3, :])
        Rul = torch.zeros(m, n, l)
        Rul[:, 0:1, :] = Lul[:, 0:1, :] + Lul[:, 1:2, :] + Lul[:, 2:3, :]
        Rul[:, 1:2, :] = l1 * Lul[:, 0:1, :] + u_l * Lul[:, 1:2, :] + l3 * Lul[:, 2:3, :]
        Rul[:, 2:3, :] = (H_l - u_l * c_l) * Lul[:, 0:1, :] + c2 * Lul[:, 1:2, :] + (H_l + u_l * c_l) * Lul[:, 2:3, :]

        return -(Rur + Rul) / (2 * self.dx)

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
n_steps = 10
DT = 0.01
tstart = 0.1


def gen_Any_data():
    data0 = any_solution(x0f,tstart)
    data_uC0 = torch.zeros(m, 3, l + igst * 2)
    tp = torch.zeros(m, 3, l + igst * 2)
    gamma = 1.4
    data_uC0[:,0,:] = data0[0,:]
    data_uC0[:,1,:] = data0[2,:]*data0[0,:]
    data_uC0[:,2,:] = data0[1,:]/(gamma-1.)+0.5*data0[0,:]*data0[2,:]*data0[2,:]
    with torch.no_grad():
        for i in range(1,n_steps+1):
            t = i*DT+tstart
            data0 = any_solution(x0f,t)
            tp[:,0,:] = data0[0,:]
            tp[:,1,:] = data0[2,:]*data0[0,:]
            tp[:,2,:] = data0[1,:]/(gamma-1.)+0.5*data0[0,:]*data0[2,:]*data0[2,:]
            data_uC0 = torch.cat((data_uC0,tp),0).float()
    torch.save(data_uC0, 'uAny.dat')


def gen_test_data():
    data_uC0 = []
    data_DT = []
    data0 = any_solution(x0f, tstart)
    uC0 = torch.zeros(m, 3, l + igst * 2)
    gamma = 1.4
    uC0[:, 0, :] = data0[0, :]
    uC0[:, 1, :] = data0[2, :] * data0[0, :]
    uC0[:, 2, :] = data0[1, :] / (gamma - 1.) + 0.5 * data0[0, :] * data0[2, :] * data0[2, :]
    ff = Roe(dx)
    ff.eval()
    with torch.no_grad():
        for i in range(n_steps + 1):
            t = i * DT + tstart
            data_uC0.append(uC0)
            data_DT.append(t)
            uC0 = runge_kutta(uC0, DT, ff)
    data_uC0 = torch.cat(data_uC0).float()
    torch.save(data_uC0, 'uRoe.dat')

def plot_u(x0, ur):
    plt.clf()
    x = to_np(torch.squeeze(x0))
    ur = to_np(torch.squeeze(ur))
    plt.clf()
    plt.scatter(x, ur)
    plt.xlim(-0.5, 0.5)
    plt.ylim(0, 1)
    plt.draw()
    plt.pause(0.1)
    
def gen_data():
    data_uC0 = []
    data_uC1 = []
    data_DT = []
    n_sample = 2000
    gamma = 1.4
    DT = 0.02
    uC0 = torch.zeros(m, 3, l + igst * 2)
    uC1 = torch.zeros(m, 3, l + igst * 2)
    with torch.no_grad():
        for i in range(n_sample):
            print(i)
            t0 = np.random.uniform(0.04, 0.08)
            data0 = any_solution(x0f, t0)
            data1 = any_solution(x0f, t0 + DT)
            uC0[:, 0, :] = data0[0, :]
            uC0[:, 1, :] = data0[2, :] * data0[0, :]
            uC0[:, 2, :] = data0[1, :] / (gamma - 1.) + 0.5 * data0[0, :] * data0[2, :] * data0[2, :]
            uC1[:, 0, :] = data1[0, :]
            uC1[:, 1, :] = data1[2, :] * data1[0, :]
            uC1[:, 2, :] = data1[1, :] / (gamma - 1.) + 0.5 * data1[0, :] * data1[2, :] * data1[2, :]
            if (i<100):
                plot_u(x0f, uC0[:, 0, :])
                plot_u(x0f, uC1[:, 0, :])
            data_uC0.append(uC0)
            data_uC1.append(uC1)
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


#gen_data()
gen_test_data()
gen_Any_data()