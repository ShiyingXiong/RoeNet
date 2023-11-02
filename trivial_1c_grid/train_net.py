import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import h5py
import torch.utils.data
import time



def cal_boundry(u, igst):
    m, n, l = u.shape
    u = torch.cat((u[:, :, -2*igst:-igst], u[:, :, igst: -igst], u[:, :, igst:2*igst]), dim=2)
    return u

def runge_kutta(z0, t1_t0, f, eps=0.01):
    n_steps = round(t1_t0 / eps)
    h = t1_t0 / n_steps
    z = z0
    for i_step in range(int(n_steps)):
        k1 = cal_boundry(z + h * f(z), igst)
        k2 = cal_boundry(0.75*z + 0.25*k1 + 0.25 * h * f(k1), igst)
        z = cal_boundry(z/3. + 2.*k2/3. + 2. * h * f(k2)/3., igst)
    return z



    
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
        self.hide = 1
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

    def cal_du(self, um):
        ul = torch.cat((um[:, :, :1], um[:, :, :-1]), 2)
        ur = torch.cat((um[:, :, 1:], um[:, :, -1:]), 2)

        data_left = torch.cat((ul, um), 1)
        data_right = torch.cat((um, ur), 1)

        lam_l = self.cal_lam(data_left).transpose(1, 2) / 10
        lam_l = torch.diag_embed(lam_l)
        lam_r = self.cal_lam(data_right).transpose(1, 2) / 10
        lam_r = torch.diag_embed(lam_r)

        L_l = self.cal_L(data_left).transpose(1, 2)
        L_l = L_l.reshape(L_l.shape[0], L_l.shape[1], self.hide, self.conp)
        L_r = self.cal_L(data_right).transpose(1, 2)
        L_r = L_r.reshape(L_r.shape[0], L_r.shape[1], self.hide, self.conp)

        R_l = torch.inverse((L_l.transpose(2,3))@L_l)@(L_l.transpose(2,3))
        R_r = torch.inverse((L_r.transpose(2,3))@L_r)@(L_r.transpose(2,3))

        um = um.transpose(1, 2).unsqueeze(-1)
        ul = ul.transpose(1, 2).unsqueeze(-1)
        ur = ur.transpose(1, 2).unsqueeze(-1)
        

        out = R_r @ (lam_r - lam_r.abs()) @ L_r @ (ur - um) + \
              R_l @ (lam_l + lam_l.abs()) @ L_l @ (um - ul)
        return -out.squeeze(-1).transpose(1, 2) / (2 * self.dx)

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
print(device)
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
m, n, l = x0.shape
x0f = torch.zeros(m, n, l + igst * 2)
x0f[:, :, 0:igst] = x0[:, :, l - igst:l] - lx
x0f[:, :, l + igst:l + igst * 2] = x0[:, :, 0:igst] + lx
x0f[:, :, igst:l + igst] = x0[:, :, 0:l]
time_size = 101


def to_np(x):
    return x.detach().cpu().numpy()

   

class Roe(nn.Module):
    def __init__(self, dx, nconp):
        super(Roe, self).__init__()
        self.dx = dx
        self.nconp = nconp
        self.L = torch.tensor([1.]).reshape([nconp,nconp])
        self.R = torch.inverse(self.L)
        self.Lam = torch.tensor([1.])

    def forward(self, u):
        ul = torch.cat((u[:, :, 0:1], u[:, :, :-1]), 2)
        ur = torch.cat((u[:, :, 1:], u[:, :, -1:]), 2)
        m, n, l = u.shape
        du = ur - u
        Rur = torch.zeros(m,n,l)
        for i in range(self.nconp):
            for j in range(self.nconp):
                for k in range(self.nconp):
                    Rur[:,k,:] = Rur[:,k,:]+(self.Lam[i]-torch.abs(self.Lam[i]))*self.L[i,j]*self.R[k,i]*du[:,j,:]
        du = u - ul
        Rul = torch.zeros(m,n,l)
        for i in range(self.nconp):
            for j in range(self.nconp):
                for k in range(self.nconp):
                    Rul[:,k,:] = Rul[:,k,:]+(self.Lam[i]+torch.abs(self.Lam[i]))*self.L[i,j]*self.R[k,i]*du[:,j,:]
        return -(Rur + Rul) / (2 * self.dx)




def any_solution(grid_size,t):
    igst = 10
    xs = -0.5
    xe = 0.5
    lx = xe - xs
    x0 = torch.tensor(range(grid_size), dtype=torch.float32, requires_grad=True).unsqueeze(0).unsqueeze(0) * lx / grid_size + xs
    x0f = torch.zeros(1, 1, grid_size + igst * 2)
    x0f[:, :, 0:igst] = x0[:, :, grid_size - igst:grid_size] - lx
    x0f[:, :, grid_size + igst:grid_size + igst * 2] = x0[:, :, 0:igst] + lx
    x0f[:, :, igst:grid_size + igst] = x0[:, :, 0:grid_size]
    u = torch.zeros(1,1,grid_size + igst * 2)
    for index in range(grid_size + igst * 2):
        tp = (x0f[0,0,index] - t+0.5)%1-0.5
        u[0,0,index] = torch.exp(-300.*torch.pow(tp,2.))
    return u
    
def test():
    device = 'cpu'
    f_neur.load_state_dict(torch.load("model.pt", map_location=lambda storage, location: storage))
    f_neur.to(device)
    f_neur.eval()
    uF = any_solution(100,0).to(device)
    
    
    
    
    RoeNet_time = []
    RoeNet_error = []
    time_accumulate = 0
    
    f_100 = Roe(1/100,1)
    f_100.eval()
    RoeSolver_time_100 = []
    RoeSolver_error_100 = []
    time_accumulate_100 = 0
    uS_100 = any_solution(100,0).to(device)
    
    f_1000 = Roe(1/1000,1)
    f_1000.eval()
    RoeSolver_time_1000 = []
    RoeSolver_error_1000 = []
    time_accumulate_1000 = 0
    uS_1000 = any_solution(1000,0).to(device)
    
    f_2000 = Roe(1/2000,1)
    f_2000.eval()
    RoeSolver_time_2000 = []
    RoeSolver_error_2000 = []
    time_accumulate_2000 = 0
    uS_2000 = any_solution(2000,0).to(device)
    
    with torch.no_grad():
        for j in range(time_size):
            t = j*DT
            
            
            RoeNet_error.append(to_np((uF-any_solution(100,t).to(device)).abs().mean()))
            time_start = time.time()
            uF = f_neur(uF, DT)
            time_end = time.time()
            time_accumulate += time_end-time_start
            RoeNet_time.append(time_accumulate)
            
            
            RoeSolver_error_100.append(to_np((uS_100-any_solution(100,t).to(device)).abs().mean()))
            time_start = time.time()
            uS_100 = runge_kutta(uS_100, DT, f_100, 1e-2)
            time_end = time.time()
            time_accumulate_100 += time_end-time_start
            RoeSolver_time_100.append(time_accumulate_100)
            
            RoeSolver_error_1000.append(to_np((uS_1000-any_solution(1000,t).to(device)).abs().mean()))
            time_start = time.time()
            uS_1000 = runge_kutta(uS_1000, DT, f_1000, 1e-3)
            time_end = time.time()
            time_accumulate_1000 += time_end-time_start
            RoeSolver_time_1000.append(time_accumulate_1000)
            
            RoeSolver_error_2000.append(to_np((uS_2000-any_solution(2000,t).to(device)).abs().mean()))
            time_start = time.time()
            uS_2000 = runge_kutta(uS_2000, DT, f_2000, 5e-4)
            time_end = time.time()
            time_accumulate_2000 += time_end-time_start
            RoeSolver_time_2000.append(time_accumulate_2000)
            
            
            
            
        
        with open('trivial1c_time.dat','w') as f_time:
            for j in range(time_size):
                f_time.write(str(j*DT))
                f_time.write('\t\t')
                f_time.write(str(RoeNet_time[j]))
                f_time.write('\t\t')
                f_time.write(str(RoeSolver_time_100[j]))
                f_time.write('\t\t')
                f_time.write(str(RoeSolver_time_1000[j]))
                f_time.write('\t\t')
                f_time.write(str(RoeSolver_time_2000[j]))
        with open('trivial1c_error.dat','w') as f_error:
            for j in range(time_size):
                f_error.write(str(j*DT))
                f_error.write('\t\t')
                f_error.write(str(RoeNet_error[j]))
                f_error.write('\t\t')
                f_error.write(str(RoeSolver_error_100[j]))
                f_error.write('\t\t')
                f_error.write(str(RoeSolver_error_1000[j]))
                f_error.write('\t\t')
                f_error.write(str(RoeSolver_error_2000[j]))
        plt.figure(1)
        plt.subplot(2,1,1)
        plt.plot(RoeNet_time, label='RoeNet_time')
        plt.plot(RoeSolver_time_100, label='RoeSolver_time_100')
        plt.plot(RoeSolver_time_1000, label='RoeSolver_time_1000')
        plt.plot(RoeSolver_time_2000, label='RoeSolver_time_2000')
        plt.legend()
        plt.subplot(2,1,2)
        plt.plot(RoeNet_error, label='RoeNet_error')
        plt.plot(RoeSolver_error_100, label='RoeSolver_error_100')
        plt.plot(RoeSolver_error_1000, label='RoeSolver_error_1000')
        plt.plot(RoeSolver_error_2000, label='RoeSolver_error_2000')
        plt.legend()
        plt.show()
        
                
                

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
    n_epoch = 500
    optimizer = torch.optim.Adam(f_neur.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)#100
    train_data_loader = torch.utils.data.DataLoader(Dataset('train'), batch_size=32, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(Dataset('test'), batch_size=32, shuffle=True)
    
    with open('loss.dat','w') as f:
        for i in range(n_epoch):
            train_loss_mse = 0
            train_loss_l1 = 0
            train_sample = 0
            f_neur.train()
            for batch_index, data_batch in enumerate(train_data_loader):
                uC0, uC1, DT = data_batch
                uC0, uC1 = uC0.to(device), uC1.to(device)
                uN1 = f_neur(uC0, DT[0].cpu().item())
                lossmse = torch.nn.functional.mse_loss(uN1, uC1)
                lossl1 = torch.nn.functional.l1_loss(uN1, uC1)
                if(i>0):
                    optimizer.zero_grad()
                    lossmse.backward()
                    optimizer.step()
                train_loss_mse += lossmse.detach().cpu().item()
                train_loss_l1 += lossl1.detach().cpu().item()
                train_sample += 1
                
            scheduler.step()
            test_loss_mse = 0
            test_loss_l1 = 0
            test_sample = 0
            f_neur.eval()
            
            with torch.no_grad():
                for batch_index, data_batch in enumerate(test_data_loader):
                    uC0, uC1, _ = data_batch
                    uC0, uC1 = uC0.to(device), uC1.to(device)
                    uN1 = f_neur(uC0, DT[0].cpu().item())
                    lossmse = torch.nn.functional.mse_loss(uN1, uC1)
                    lossl1 = torch.nn.functional.l1_loss(uN1, uC1)
                    test_loss_mse += lossmse.detach().cpu().item()
                    test_loss_l1 += lossl1.detach().cpu().item()
                    test_sample += 1
            print(i, train_loss_mse / train_sample, test_loss_mse / test_sample)
            f.write(str(i+0.0))
            f.write('\t\t')
            f.write(str(train_loss_mse / train_sample))
            f.write('\t\t')
            f.write(str(train_loss_l1 / train_sample))
            f.write('\t\t')
            f.write(str(test_loss_mse / test_sample))
            f.write('\t\t')
            f.write(str(test_loss_l1 / test_sample))
            f.write('\t\t')
            f.write('\n')      
            torch.save(f_neur.state_dict(), "model.pt")


if __name__ == '__main__':
    #train()
    test()
