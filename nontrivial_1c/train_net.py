import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import random
import h5py
import os
import torch.utils.data
#from torch.utils.tensorboard import SummaryWriter

def to_np(x):
    return x.detach().cpu().numpy()

def cal_boundry(u, igst):
    m, n, l = u.shape
    u = torch.cat((u[:, :, -2*igst:-igst], u[:, :, igst: -igst], u[:, :, igst:2*igst]), dim=2)
    return u


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.kernel_size = kernel_size
        self.left = nn.Sequential(
            nn.Conv1d(inchannel, outchannel, kernel_size=kernel_size, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(outchannel, outchannel, kernel_size=1, padding=0),
        )
        self.shortcut = nn.Sequential()
        if inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(inchannel, outchannel, kernel_size=1, padding=0),
            )

    def forward(self, x):
        if self.kernel_size == 3:
            out = self.left(F.pad(x, pad=[1, 1], mode='replicate'))
        else:
            out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ODEnet(nn.Module):
    def __init__(self, dx, igst, eps):
        super(ODEnet, self).__init__()
        self.conp = 1
        self.hide = 4
        self.igst = igst
        self.dx = dx
        self.eps = eps
        
        self.cal_lam = nn.Sequential(ResidualBlock(self.conp*2, 16),
                                      ResidualBlock(16, 16),
                                      ResidualBlock(16, 32),
                                      ResidualBlock(32, 64),
                                      ResidualBlock(64, 64),
                                      ResidualBlock(64, 64, kernel_size=1),
                                      nn.Conv1d(64, self.hide, kernel_size=1, padding=0),
                                      )
        self.cal_L = nn.Sequential(ResidualBlock(self.conp*2, 16),
                                    ResidualBlock(16, 16),
                                    ResidualBlock(16, 32),
                                    ResidualBlock(32, 64),
                                    ResidualBlock(64, 64),
                                    ResidualBlock(64, 64, kernel_size=1),
                                    nn.Conv1d(64, self.conp*self.hide, kernel_size=1, padding=0),
                                    )
        # self.cal_R = nn.Sequential(ResidualBlock(self.conp*2, 16),
        #                             ResidualBlock(16, 16),
        #                             ResidualBlock(16, 32),
        #                             ResidualBlock(32, 64),
        #                             ResidualBlock(64, 64),
        #                             ResidualBlock(64, 64, kernel_size=1),
        #                             nn.Conv1d(64, self.conp*self.hide, kernel_size=1, padding=0),
        #                             )

    def cal_du(self, um):
        ul = torch.cat((um[:, :, -1:], um[:, :, :-1]), 2)
        ur = torch.cat((um[:, :, 1:], um[:, :, :1]), 2)

        data_left = torch.cat((ul, um), 1)
        data_right = torch.cat((um, ur), 1)

        lam_l = self.cal_lam(data_left).transpose(1, 2)# / 10
        lam_l = torch.diag_embed(lam_l)
        lam_r = self.cal_lam(data_right).transpose(1, 2)# / 10
        lam_r = torch.diag_embed(lam_r)

        L_l = self.cal_L(data_left).transpose(1, 2)
        L_l = L_l.reshape(L_l.shape[0], L_l.shape[1], self.hide, self.conp)
        L_r = self.cal_L(data_right).transpose(1, 2)
        L_r = L_l.reshape(L_r.shape[0], L_r.shape[1], self.hide, self.conp)

        # R_l = self.cal_R(data_left).transpose(1, 2)
        # R_l = R_l.reshape(R_l.shape[0], R_l.shape[1], self.conp, self.hide)
        # R_r = self.cal_R(data_right).transpose(1, 2)
        # R_r = R_l.reshape(R_r.shape[0], R_r.shape[1], self.conp, self.hide)
        R_l = torch.inverse((L_l.transpose(2,3))@L_l)@(L_l.transpose(2,3))
        R_r = torch.inverse((L_r.transpose(2,3))@L_r)@(L_r.transpose(2,3))
        um = um.transpose(1, 2).unsqueeze(-1)
        ul = ul.transpose(1, 2).unsqueeze(-1)
        ur = ur.transpose(1, 2).unsqueeze(-1)
        

        out = R_r @ (lam_r - lam_r.abs()) @ L_r @ (ur - um) + \
              R_l @ (lam_l + lam_l.abs()) @ L_l @ (um - ul)
        
        #out = (lam_r - lam_r.abs()) @ (ur - um) + \
        #      (lam_l + lam_l.abs()) @ (um - ul)
        return -out.squeeze(-1).transpose(1, 2) / (2 * self.dx)

    def forward(self, z0, t1_t0):
        # return z0 + self.cal_du(z0) * t1_t0
        n_steps = round(t1_t0 / self.eps)
        h = t1_t0 / n_steps
        z = z0
        for i_step in range(int(n_steps)):
            # k1 = z + h * self.cal_du(z)
            # k1 = cal_boundry(k1, igst)

            # k2 = 0.75 * z + 0.25 * k1 + 0.25 * h * self.cal_du(k1)
            # k2 = cal_boundry(k2, igst)

            # z = z / 3. + 2 * k2 / 3. + 2. * h * self.cal_du(k2) / 3.
            # z = cal_boundry(z, igst)
            z = cal_boundry(z + h * self.cal_du(z), self.igst)
        return z


device = 'cuda:0'
device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
igst = 10
grid_size = 100
time_size = 80
xs = -0.5
xe = 0.5
lx = xe - xs
dx = lx / grid_size
DT = 0.01
f_neur = ODEnet(dx, igst, eps=0.001)
f_neur.to(device)
x0 = torch.tensor(range(grid_size), dtype=torch.float32, requires_grad=True).unsqueeze(0).unsqueeze(0) * lx / grid_size + xs
m, n, l = x0.shape
x0f = torch.zeros(m, n, l + igst * 2)
x0f[:, :, 0:igst] = x0[:, :, l - igst:l] - lx
x0f[:, :, l + igst:l + igst * 2] = x0[:, :, 0:igst] + lx
x0f[:, :, igst:l + igst] = x0[:, :, 0:l]

def to_np(x):
    return x.detach().cpu().numpy()


def plot_u(x0, ur, un):
    plt.clf()
    x = to_np(torch.squeeze(x0))
    un = to_np(un)
    ur = to_np(ur)
    plt.clf()
    plt.scatter(x, ur, c="g")
    plt.scatter(x, un)
    plt.xlim(-0.5, 0.5)
    plt.ylim(-1., 2.)
    plt.draw()
    plt.pause(1)

def test():
    f_neur.load_state_dict(torch.load("model.pt", map_location=lambda storage, location: storage))
    f_neur.to(device)
    f_neur.eval()
    uR = torch.load('uRoe.dat')
    uF = uR[0:1, :, :].to(device)
    check_dim = 0
    with torch.no_grad():
        with open('nontrivial1c.dat','w') as f:
            for j in range(time_size):
                f.write('ZONE T = "zone')
                f.write('\t\t')
                f.write(str(j))
                f.write('"')
                f.write('\n')
                for i in range(grid_size+20):
                    tnp = to_np(x0f)
                    f.write(str(tnp[0,0,i]))
                    f.write('\t\t')
                    rnp = to_np(uR)
                    f.write(str(rnp[j,0,i]))
                    f.write('\t\t')
                    fnp = to_np(uF)
                    f.write(str(fnp[0,0,i]))
                    f.write('\n')
                uF = f_neur(uF, DT)
        uF = uR[0:1, :, :].to(device)
        for i in range(time_size):
            # print(uF.shape,uA.shape)
            plot_u(x0f, uR[i, check_dim, :], uF[:, check_dim, :])
            uF = f_neur(uF, DT)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_type):
        f = h5py.File('data.h5')
        self.uC0 = f['uC0'][:]
        self.uC1 = f['uC1'][:]
        self.DT = f['DT'][:]
        split = int(self.uC0.shape[0] * 0.9)
        if data_type == 'train':
            self.uC0, self.uC1, self.DT = self.uC0[:split], self.uC1[:split], self.DT[:split]
        else:
            self.uC0, self.uC1, self.DT = self.uC0[split:], self.uC1[split:], self.DT[split:]
        f.close()

    def __getitem__(self, index):
        return self.uC0[index], self.uC1[index], self.DT[index]

    def __len__(self):
        return self.uC0.shape[0]


def train():
    
    #writer = SummaryWriter('runs/roe')
    n_epoch = 1000
    #loss_func = torch.nn.functional.l1_loss
    optimizer = torch.optim.Adam(f_neur.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    train_data_loader = torch.utils.data.DataLoader(Dataset('train'), batch_size=32, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(Dataset('test'), batch_size=32, shuffle=True)
    lowest_test_loss = 99999
    for i in range(n_epoch):
        train_loss = 0
        train_sample = 0
        f_neur.train()
        for batch_index, data_batch in enumerate(train_data_loader):
            uC0, uC1, DT = data_batch
            uC0, uC1 = uC0.to(device), uC1.to(device)
            uN1 = f_neur(uC0, DT[0].cpu().item())
            #weight = F.pad(uC1, pad=[1, 1], mode='replicate')
            #weight = weight[:, :, 2:] - weight[:, :, :-2]
            #weight = weight.norm(dim=1, keepdim=True) * 10 + 0.1
            #loss = (((uN1 - uC1) * weight)**2).mean()
            loss = torch.nn.functional.mse_loss(uN1, uC1)
            #loss = torch.nn.functional.l1_loss(uN1, uC1)
            optimizer.zero_grad()
            loss.backward()
            train_loss += loss.detach().cpu().item()
            train_sample += 1
            optimizer.step()
        scheduler.step()
        test_loss = 0
        test_sample = 0
        f_neur.eval()
        with torch.no_grad():
            for batch_index, data_batch in enumerate(test_data_loader):
                uC0, uC1, _ = data_batch
                uC0, uC1 = uC0.to(device), uC1.to(device)
                uN1 = f_neur(uC0, DT[0].cpu().item())
                #weight = F.pad(uC1, pad=[1, 1], mode='replicate')
                #weight = weight[:, :, 2:] - weight[:, :, :-2]
                #weight = weight.norm(dim=1, keepdim=True) * 10 #+ 0.1
                #loss = (((uN1 - uC1) * weight) ** 2).mean()
                #loss = torch.nn.functional.mse_loss(uN1, uC1)
                loss = torch.nn.functional.l1_loss(uN1, uC1)
                test_loss += loss.detach().cpu().item()
                test_sample += 1
        print(i + 1, train_loss / train_sample, test_loss / test_sample)
        #writer.add_scalars('loss', {'train': train_loss / train_sample,
        #                            'test': test_loss / test_sample}, i + 1)
        #writer.flush()
        if lowest_test_loss > test_loss / test_sample:
            torch.save(f_neur.state_dict(), "model.pt")
            lowest_test_loss = test_loss / test_sample
    #writer.close()


if __name__ == '__main__':
    train()
    test()
