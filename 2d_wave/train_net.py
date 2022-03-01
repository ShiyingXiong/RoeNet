import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import torch.nn.functional as F
import torch.nn as nn
import random
import h5py
import os
import torch.utils.data
#from torch.utils.tensorboard import SummaryWriter

from gen_data import any_solution

def cal_boundry(u, igst):
    u = torch.cat((u[:, :, -2*igst:-igst,:], u[:, :, igst: -igst,:], u[:, :, igst:2*igst,:]), dim=2)
    u = torch.cat((u[:, :, :, -2*igst:-igst], u[:, :, :, igst: -igst], u[:, :, :, igst:2*igst]), dim=3)
    return u

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.kernel_size = kernel_size
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size, padding=(kernel_size-1)//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=1, padding=0),
        )
        self.shortcut = nn.Sequential()
        if inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, padding=0),
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ODEnet(nn.Module):
    def __init__(self, dx, igst, eps):
        super(ODEnet, self).__init__()
        self.conp = 1
        self.hide = 8
        self.igst = igst
        self.dx = dx
        self.eps = eps

        self.cal_lam_v = nn.Sequential(ResidualBlock(self.conp*2, 16),
                                      ResidualBlock(16, 16),
                                      ResidualBlock(16, 32),
                                      ResidualBlock(32, 64),
                                      ResidualBlock(64, 64),
                                      ResidualBlock(64, 64, kernel_size=1),
                                      nn.Conv2d(64, self.hide, kernel_size=1, padding=0),
                                      )
        self.cal_lam_h = nn.Sequential(ResidualBlock(self.conp*2, 16),
                                      ResidualBlock(16, 16),
                                      ResidualBlock(16, 32),
                                      ResidualBlock(32, 64),
                                      ResidualBlock(64, 64),
                                      ResidualBlock(64, 64, kernel_size=1),
                                      nn.Conv2d(64, self.hide, kernel_size=1, padding=0),
                                      )
        self.cal_L_v = nn.Sequential(ResidualBlock(self.conp*2, 16),
                                   ResidualBlock(16, 16),
                                   ResidualBlock(16, 32),
                                   ResidualBlock(32, 64),
                                   ResidualBlock(64, 64),
                                   ResidualBlock(64, 64, kernel_size=1),
                                   nn.Conv2d(64, self.conp*self.hide, kernel_size=1, padding=0),
                                   )        
        self.cal_L_h = nn.Sequential(ResidualBlock(self.conp*2, 16),
                                   ResidualBlock(16, 16),
                                   ResidualBlock(16, 32),
                                   ResidualBlock(32, 64),
                                   ResidualBlock(64, 64),
                                   ResidualBlock(64, 64, kernel_size=1),
                                   nn.Conv2d(64, self.conp*self.hide, kernel_size=1, padding=0),
                                   )

    def cal_du(self, um):
        ul = torch.cat((um[:, :, :, :1], um[:, :, :, :-1]), 3)
        ur = torch.cat((um[:, :, :, 1:], um[:, :, :, -1:]), 3)
        ut = torch.cat((um[:, :, :1, :], um[:, :, :-1, :]), 2)
        ub = torch.cat((um[:, :, 1:, :], um[:, :, -1:, :]), 2)  # pad boundaries? 
        # print(ul.shape); print(ur.shape); print(ub.shape); print(ut.shape); 

        data_left = torch.cat((ul, um), 1)
        data_right = torch.cat((um, ur), 1)
        data_top = torch.cat((ut, um), 1)
        data_bottom = torch.cat((um, ub), 1)

        # print(data_left.shape); print(data_right.shape); print(data_top.shape); print(data_bottom.shape); 

        # lam_l, lam_r, lam_b, lam_t, batch * hide * height * width 
        # -> batch * height * width * hide - >  batch * height * width * diag_matrix
        lam_l = self.cal_lam_h(data_left).transpose(1, 2).transpose(2, 3)  / 10 
        lam_l = torch.diag_embed(lam_l)
        lam_r = self.cal_lam_h(data_right).transpose(1, 2).transpose(2, 3)  / 10
        lam_r = torch.diag_embed(lam_r)  
        lam_t = self.cal_lam_v(data_top).transpose(1, 2).transpose(2, 3)  / 10 
        lam_t = torch.diag_embed(lam_t)
        lam_b = self.cal_lam_v(data_bottom).transpose(1, 2).transpose(2, 3)  / 10
        lam_b = torch.diag_embed(lam_b)  

        # L_l, L_r, L_b, L_t, batch * conp * height * width 
        # -> batch * height * width * hide * conp
        L_l = self.cal_L_h(data_left).transpose(1, 2).transpose(2, 3)
        L_l = L_l.reshape(L_l.shape[0], L_l.shape[1], L_l.shape[2], self.hide, self.conp)
        L_r = self.cal_L_h(data_right).transpose(1, 2).transpose(2, 3)
        L_r = L_r.reshape(L_r.shape[0], L_r.shape[1], L_r.shape[2], self.hide, self.conp)        
        L_t = self.cal_L_v(data_top).transpose(1, 2).transpose(2, 3)
        L_t = L_t.reshape(L_t.shape[0], L_t.shape[1], L_t.shape[2], self.hide, self.conp)
        L_b = self.cal_L_v(data_bottom).transpose(1, 2).transpose(2, 3)
        L_b = L_b.reshape(L_b.shape[0], L_b.shape[1], L_b.shape[2], self.hide, self.conp)

        R_l = torch.inverse((L_l.transpose(3,4))@L_l)@(L_l.transpose(3,4))
        R_r = torch.inverse((L_r.transpose(3,4))@L_r)@(L_r.transpose(3,4))
        R_t = torch.inverse((L_t.transpose(3,4))@L_t)@(L_t.transpose(3,4))
        R_b = torch.inverse((L_b.transpose(3,4))@L_b)@(L_b.transpose(3,4))

        um = um.transpose(1, 2).transpose(2, 3).unsqueeze(-1)
        ul = ul.transpose(1, 2).transpose(2, 3).unsqueeze(-1)
        ur = ur.transpose(1, 2).transpose(2, 3).unsqueeze(-1)
        ut = ut.transpose(1, 2).transpose(2, 3).unsqueeze(-1)
        ub = ub.transpose(1, 2).transpose(2, 3).unsqueeze(-1)
        

        out = R_r @ (lam_r - lam_r.abs()) @ L_r @ (ur - um) + \
              R_l @ (lam_l + lam_l.abs()) @ L_l @ (um - ul) + \
              R_t @ (lam_t + lam_t.abs()) @ L_t @ (ut - um) + \
              R_b @ (lam_b + lam_b.abs()) @ L_b @ (um - ub) 

        return -out.squeeze(-1).transpose(2, 3).transpose(1, 2) / (2 * self.dx)

    def forward(self, z0, t1_t0):
        # return z0 + self.cal_du(z0) * t1_t0
        n_steps = round(t1_t0 / self.eps)
        h = t1_t0 / n_steps
        z = z0
        for i_step in range(int(n_steps)):
            z = cal_boundry(z + h * self.cal_du(z), self.igst)
        return z


device = 'cuda:0'
device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
igst = 10
grid_size = 100
xs = -0.5
xe = 0.5
lx = xe - xs
dx = lx / grid_size
DT = 0.02
f_neur = ODEnet(dx, igst, eps=0.01)
f_neur.to(device)
x0 = torch.tensor(range(grid_size), dtype=torch.float32, requires_grad=True).unsqueeze(0).unsqueeze(0) * lx / grid_size + xs
y0 = torch.tensor(range(grid_size), dtype=torch.float32, requires_grad=True).unsqueeze(0).unsqueeze(0) * lx / grid_size + xs
m, n, l = x0.shape
x0f = x0
y0f = y0
# x0f = torch.zeros(m, n, l + igst * 2)
# x0f[:, :, 0:igst] = x0[:, :, l - igst:l] - lx
# x0f[:, :, l + igst:l + igst * 2] = x0[:, :, 0:igst] + lx
# x0f[:, :, igst:l + igst] = x0[:, :, 0:l]
time_size = 51
def to_np(x):
    return x.detach().cpu().numpy()

def plot_u(x0, y0, ur, idx):
    fig = plt.figure()
    ax = Axes3D(fig)
    x = to_np(torch.squeeze(x0))
    y = to_np(torch.squeeze(y0))
    ur = to_np(ur)
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y,ur)
    plt.savefig('results/'+str(idx)+'.jpg')


def plot_more_u(x0, ur_l, un_l, ut_l):
    for i in range(len(ur_l)):
        x = to_np(torch.squeeze(x0))
        un = to_np(un_l[i])
        ut = to_np(ut_l[i])
        ur = to_np(ur_l[i])
        plt.scatter(x, ur, c="g")
        plt.scatter(x, un,c='c')
        plt.plot(x, ut, c="red")
        plt.xlim(-0.5, 0.5)
        plt.ylim(-0.2, 1.2)
    plt.show()
    
def test():
    igst = 10
    grid_size = 100
    xs = -0.6
    xe = 0.6
    lx = xe - xs
    dx = lx / grid_size
    x0 = torch.tensor(range(grid_size+igst*2), dtype=torch.float32, requires_grad=True).unsqueeze(0).unsqueeze(0) * lx / (grid_size+igst*2) + xs
    y0 = torch.tensor(range(grid_size+igst*2), dtype=torch.float32, requires_grad=True).unsqueeze(0).unsqueeze(0) * lx / (grid_size+igst*2) + xs
    m, n, l = x0.shape
    x0f = x0
    y0f = y0

    device = 'cpu'
    f_neur.load_state_dict(torch.load("model.pt", map_location=lambda storage, location: storage))
    f_neur.to(device)
    f_neur.eval()
    # uR = torch.load('uRoe.dat')
    # uA = torch.load('uAny.dat')
    # uF = uR[0:1, :, :].to(device)
    check_dim = 0

    t0 = 0.0
    uC0 = cal_boundry(any_solution(x0f,y0f,t0).unsqueeze(0), igst)
    uF = uC0
    

    with torch.no_grad():
        # with open('trivial1c.dat','w') as f:
        #     ferr = open('err0.dat','w')
        #     for j in range(time_size):
        #         f.write('ZONE T = "zone')
        #         f.write('\t\t')
        #         f.write(str(j))
        #         f.write('"')
        #         f.write('\n')
        #         for i in range(grid_size+20):
        #             tnp = to_np(x0f)
        #             f.write(str(tnp[0,0,i]))
        #             f.write('\t\t')
        #             fnp = to_np(uF)
        #             f.write(str(fnp[0,0,i]))
        #             f.write('\t\t')
        #             rnp = to_np(uR)
        #             f.write(str(rnp[j,0,i]))
        #             f.write('\t\t')
        #             anp = to_np(uA)
        #             f.write(str(anp[j,0,i]))
        #             f.write('\n')
        #         print(j,(uF[0,0,:]-uA[j,0,:]).abs().mean(),(uR[j,0,:]-uA[j,0,:]).abs().mean())
        #         ferr.write(str(j*DT))
        #         ferr.write('\t\t')
        #         ferr.write(str(to_np((uF[0,0,:]-uA[j,0,:]).abs().mean())))
        #         ferr.write('\t\t')
        #         ferr.write(str(to_np((uR[j,0,:]-uA[j,0,:]).abs().mean())))
        #         ferr.write('\n')
        #         uF = f_neur(uF, DT)
        #     ferr.close
        # uF = uR[0:1, :, :].to(device)


        # ur_l = []
        # un_l = [] 
        # ut_l = [] 
        uf_3d = [uF[0, check_dim, :, :].numpy()]
        time_3d = [0]
        for i in range(time_size):
            plot_u(x0f, y0f, uF[0, check_dim, :, :], i)
            uF = f_neur(uF, DT)
            # print(uF.shape)
            uf_3d.append(uF[0, check_dim, :, :].numpy())
            # print(len(uf_3d))
            time_3d.append((i+1)*DT)


        # plot_more_u(x0f, ur_l, un_l, ut_l)
        # print(time_3d)

        # X,Y = np.meshgrid(x0f,time_3d)
        # Z = np.stack(uf_3d, axis=0)
        # print(Z.shape)

        # X = X[:, 10:-10]
        # Y = Y[:, 10:-10]
        # Z = Z[:, 10:-10]

        # fig = plt.figure()
        # ax = Axes3D(fig)
        # print(X.shape)
        # training_period = (int)(0.2 / DT) + 1
        # ax.plot_surface(X[:training_period,:],Y[:training_period,:],Z[:training_period,:],rstride=1,cstride=1,alpha=0.8)
        # ax.plot_surface(X[training_period:,:],Y[training_period:,:],Z[training_period:,:],rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'),alpha=0.6,zorder=100)
        # ax.set_xlabel('$x$')
        # ax.set_ylabel('$t$')
        # ax.set_zlabel('predicted $u$')
        # plt.show() 


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
    optimizer = torch.optim.Adam(f_neur.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)#32
    train_data_loader = torch.utils.data.DataLoader(Dataset('train'), batch_size=8, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(Dataset('test'), batch_size=8, shuffle=True)
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
        test_loss_2 = 0
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
                loss2 = torch.nn.functional.mse_loss(uN1, uC1)
                test_loss += loss.detach().cpu().item()
                test_loss_2 += loss2.detach().cpu().item()
                test_sample += 1
        print(i + 1, train_loss / train_sample, test_loss / test_sample, test_loss_2/test_sample)
        #writer.add_scalars('loss', {'train': train_loss / train_sample,
        #                            'test': test_loss / test_sample}, i + 1)
        #writer.flush()
        if lowest_test_loss > test_loss / test_sample:
            torch.save(f_neur.state_dict(), "model.pt")
            lowest_test_loss = test_loss / test_sample
    #writer.close()


if __name__ == '__main__':
    # train()
    test()
