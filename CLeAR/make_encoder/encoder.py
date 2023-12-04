import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle

#@ setting
task_name = 'cycle_navigation' #'modular_arithmetic', 'parity_check', 'even_pairs', 'cycle_navigation'
train_step = 10_000
epochs = 10
batch_size = 128
binarize_setting = 0
in_dim = 3
rand_dim = 3## Random dim fix to 3
out_dim = 3

'''
binarize_setting: determine wheter use 1/0 or 1/-1
0: use 0/1
1: use 1/-1
'''
class Net(nn.Module):
    def __init__(self, in_dim, rand_dim, out_dim, binarize_setting):
        super().__init__()
        self.binarize_setting = binarize_setting

        self.enc1 = nn.Linear(in_dim+rand_dim, 128)
        self.enc2 = nn.Linear(128, 128)
        self.enc3 = nn.Linear(128, out_dim)
        self.dec1 = nn.Linear(out_dim, 128)
        self.dec2 = nn.Linear(128, 128)
        self.dec3 = nn.Linear(128, in_dim)

        self.activation = nn.ReLU()

        '''
        for m in self.modules():
            print('g',m)
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, mean=0, std = 0.1)
        '''


    def binarize(self, x):
        x_detach = x.clone().detach()
        if self.binarize_setting==0:
            x = torch.where(x>0.5, x+(1-x_detach), x - x_detach)
        elif self.binarize_setting==1:
            x = torch.where(x > 0.0, x + (1 - x_detach), x - (x_detach+1))
        return x

    def sig_tanh(self, x):
        if self.binarize_setting==0:
            x = F.sigmoid(x)
        elif self.binarize_setting==1:
            x = F.tanh(x)
        return x

    def forward(self, x):
        x_out = self.activation(self.enc1(x))
        x_out = self.activation(self.enc2(x_out))
        x_out = self.enc3(x_out)
        x_out = self.binarize(x_out)
        x_recon = self.activation(self.dec1(x_out))
        x_recon = self.activation(self.dec2(x_recon))
        x_recon = self.dec3(x_recon)
        x_recon_b = self.binarize(x_recon)
        x_recon_s = self.sig_tanh(x_recon)
        return x_out, x_recon_b, x_recon_s

    def eval_(self):
        data_tmp = torch.zeros((3, 6))
        data_tmp[:,:in_dim] = torch.eye(in_dim)
        x_out = self.activation(self.enc1(data_tmp))
        x_out = self.activation(self.enc2(x_out))
        x_out = self.enc3(x_out)
        x_out = self.binarize(x_out)

        tmp = torch.zeros((8, 3))
        for num in range(8):
            tmp_num = num
            for idx in range(3):
                tmp[num,idx] = tmp_num%2
                tmp_num = tmp_num //2
        if binarize_setting ==1:
            tmp = tmp*2-1
        x_recon = self.activation(self.dec1(tmp))
        x_recon = self.activation(self.dec2(x_recon))
        x_recon = self.dec3(x_recon)
        x_recon = self.binarize(x_recon)
        return x_out, x_recon




def run(task_name, train_step, binarize_setting, in_dim, rand_dim,out_dim, batch_size, epochs):
    running_loss = [.0, .0, .0, .0, .0, .0]
    #loader = task_loader(task_name, train_step, batch_size)
    with open(file='data.pickle', mode='rb') as f:
        data = pickle.load(f)
    model = Net(in_dim, rand_dim, out_dim, binarize_setting)
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)
    loss_recons = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005, )
    if binarize_setting==0:
        mean_out = 0.5
    elif binarize_setting==1:
        mean_out = 0.0

    parm1 = 0.0
    for epoch in range(epochs):
        for step in range(train_step):  # loop over the dataset multiple times
            input_, length = data[step]
            #print('1',input_.shape) #(128, 7, 3)
            input_ = torch.from_numpy(input_)
            rand_pad = torch.randint(2,(128,length,rand_dim))
            input_noise = torch.cat((input_,rand_pad), dim=2)
            if binarize_setting == 1:
                input_ = input_*2-1
                input_noise = input_noise*2-1
            #print('2',input_noise.shape) #torch.Size([128, 7, 6])
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            x_out, x_recon_b, x_recon_s = model(input_noise)
            #print('3',x_out.shape) #torch.Size([128, 7, 3])
            #print('4',x_recon.shape) #torch.Size([128, 7, 3])
            x_out_flatten = torch.flatten(x_out, end_dim =1)
            #print('5',x_out_flatten.shape) #torch.Size([896, 3])

            loss_recon_b = torch.mean(torch.abs(x_recon_b-input_))
            #loss_recon_s = loss_recons(x_recon_s,input_)
            loss_recon = loss_recon_b

            if loss_recon ==0.0 :
                parm1 = 1

            loss_set_mean = torch.mean(torch.abs(torch.mean(x_out, (0,1))-mean_out))
            loss_cov_nondiag = torch.mean(torch.abs(torch.cov(x_out_flatten)-torch.diagonal(torch.cov(x_out_flatten))))
            loss_cov_diag = torch.mean(torch.abs(torch.diagonal(torch.cov(x_out_flatten))-((1-mean_out)**2)))
            loss = 50*loss_recon\
                   +parm1*loss_set_mean\
                   +parm1*loss_cov_nondiag\
                   +parm1*loss_cov_diag
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss[0] += loss.item()
            running_loss[1] += loss_recon.item()
            running_loss[2] += loss_set_mean.item()
            running_loss[3] += loss_cov_nondiag.item()
            running_loss[4] += loss_cov_diag.item()
            term = 20
            if step % term == term-1:  # print every 2000 mini-batches
                print(f'[{epoch} epoch {step + 1}] loss: {running_loss[0] / term:.3f}'
                      f'      loss_recon: {running_loss[1] / term:.3f}'
                      f'    loss_set_mean: {running_loss[2] / term:.3f}'
                      f'    loss_cov_nondiag: {running_loss[3] / term:.3f}'
                      f'    loss_cov_diag: {running_loss[4] / term:.3f}')
                if running_loss[1] == 0.0:
                    print(model.eval_())
                running_loss = [.0, .0, .0, .0, .0, .0]

run(task_name, train_step, binarize_setting, in_dim, rand_dim,out_dim, batch_size, epochs)
'''
enc1.weight torch.Size([3, 6])
enc1.bias torch.Size([3])
dec1.weight torch.Size([3, 3])
dec1.bias torch.Size([3])
dec2.weight torch.Size([3, 3])
dec2.bias torch.Size([3])
'''

