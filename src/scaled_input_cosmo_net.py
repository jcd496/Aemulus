import torch
import sys
import os
import numpy as np
from time import monotonic
from torch import nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
from torch.autograd import Function
HOD=8
LABEL=10+7
ERROR = 11+7
NPL = 1024#16  #neurons per layer
#data class.  loads data from csv file which is stored column wise.  column 0-7 hod training params, column 8 radius training param,
#column 9 regression target, column 10 error on data point
class DataClass(Dataset):

    def __init__(self, csv_file):
        self.file = np.genfromtxt(csv_file, delimiter=",")
        self.labels = self.file[:,16]
        self.inputs = self.file[:,:16]
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        labels = self.labels[idx]
        inputs = self.inputs[idx]
        sample = {'inputs' : inputs , 'label': labels}
        return sample

#branch neural net.  causes radius parameter to "feel" more effect of loss function and reduces computation
class BranchNet(nn.Module):
    def __init__(self):
        super(BranchNet, self).__init__()
        #self.cosmo_fc1 = nn.Linear(COSMO,7)
        #self.cosmo_fc2 = nn.Linear(7,1)
            
        self.hod_fc1 = nn.Linear(16,NPL)
        self.hod_fc2 = nn.Linear(NPL,NPL)
        self.hod_fc3 = nn.Linear(NPL,NPL//2)  #change 7 to 256 in each of above
        self.hod_fc4 = nn.Linear(NPL//2,1)  #change 7 to 256 in each of above
            
        #self.combo_fc1 = nn.Linear(2, NPL) #3,256
        #self.combo_fc2 = nn.Linear(NPL,1)  #256,1


    def forward(self, x):
        """cosmo_out = self.cosmo_fc1(x['cosmo_params'].float())
        cosmo_out = torch.tanh(cosmo_out)
        cosmo_out = self.cosmo_fc2(cosmo_out)"""

        hod_out = self.hod_fc1(x.float())
        hod_out = torch.tanh(hod_out)
        hod_out = self.hod_fc2(hod_out)
        hod_out = torch.tanh(hod_out)
        hod_out = self.hod_fc3(hod_out)
        hod_out = torch.tanh(hod_out)
        hod_out = self.hod_fc4(hod_out)
        return hod_out

#keep training time statistics
class timer():
    def __init__(self):
        self.count=0
        self.sum=0.0
        self.avg=0.0
    def update(self,time):
        self.sum+=time
        self.count+=1
    def average(self):
        return self.sum/self.count


# train off of chisquare loss function to encapsulate error on training data
def ChiSquare(outputs, labels, errors):
    diff = torch.add(outputs,-1,labels)
    power=torch.pow(diff,2)
    error_sq=torch.pow(errors,2)
    ratio = torch.div(power,error_sq)
    loss = torch.sum(ratio)/len(outputs)
    return loss, ratio

#train network and keep statistics
def train(model, optimizer, loss_func, data_loader, epochs, device):
    start_time = monotonic()
    criterion = nn.MSELoss()
    epoch_time = timer()
    load_time = timer()
    compute_time = timer()
    running_loss=0.0
    outputs =None
    loss_per_epoch=[]
    for epoch in range(epochs):
        epoch_time_s = monotonic()
        running_loss = 0.0
        load_time_s = monotonic()
        for i, data in enumerate(data_loader, 0):
            #load and keep stats for c2 and c3
            inputs = data['inputs']
            labels = data['label']
            
            labels=labels.float()
            inputs, labels = inputs.to(device), labels.to(device) 
            #inputs=inputs.view(-1,1)
            labels = labels.view(-1,1)

            #inputs, labels = inputs.to(device), labels.to(device)
            load_time_f = monotonic()
            

            #feed forward, calc gradients, update params and keep stats for c2
            comp_time_s = monotonic()			
            optimizer.zero_grad()
            outputs = model(inputs)


            loss = criterion(outputs, labels)
            
            loss.backward()
            
            optimizer.step()	
            comp_time_f = monotonic()
            
            #stats, running loss
            load_time.update((load_time_f-load_time_s))
            compute_time.update((comp_time_f-comp_time_s))
            running_loss+=loss.item()
        
        loss_per_epoch.append(running_loss)
        epoch_time_f = monotonic()
        epoch_time.update((epoch_time_f-epoch_time_s))
        print("epoch:", epoch, "loss:", running_loss)
    end_time = monotonic()
    labels, outputs = labels.to(torch.device('cpu')), outputs.to(torch.device('cpu'))
    labels = labels.detach().numpy()
    outputs = outputs.detach().numpy()
    print("Total Time: {:.3f}    Num Epochs: {:d}   Average Epoch: {:.3f}   Average Load: {:.3f}    Average FWD+BWD: {:.3f}    Final Loss: {:.3f}".format(end_time-start_time, epochs, epoch_time.average(),load_time.average(), compute_time.average(), running_loss))
    return outputs, labels, loss_per_epoch

#test network and report statistics
def test(model, loss_func, data_loader, size, device):
    test_loss = 0.0
    Labels = None
    output = None
    
    criterion = nn.MSELoss()
    fractional_error=np.zeros((size,1),dtype=float) 
    for i, data in enumerate(data_loader, 0):
            inputs = data['inputs']
            labels = data['label']
            labels=labels.float()
            labels = labels.view(-1,1)
            inputs, labels = inputs.to(device), labels.to(device) 
            #model.eval()
            output = model(inputs)

            test_loss+=criterion(output, labels)
            print("test loss:",test_loss.item())
            labels, output = labels.to(torch.device('cpu')), output.to(torch.device('cpu'))
            fractional_error+=abs(output.detach().numpy()-labels.detach().numpy())/labels.detach().numpy()*100
    labels = labels.detach().numpy()
    output = output.detach().numpy()
    return labels, output, test_loss, fractional_error



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='emulator net')
    parser.add_argument('--save_path', type=str, default=None ,help='path to save dictionary of optimizer and model state')
    parser.add_argument('--load_path', type=str, default=None, help='load state dictionary, try sicn_24000.rprop.pt')
    parser.add_argument('--loss_func', type=str, default='chi-squared', help='loss function to train off, chi-squared or MSE')
    parser.add_argument('--use_cuda', type=bool, default=False, help='Use CUDA if available')
    parser.add_argument('--workers', type=int, default=0, help='Number of dataloader workers')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train over')
    args=parser.parse_args()

    device = None
    if args.use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print ('cuda_available:', torch.cuda.is_available())
    print ('device:', device)

    
    training_data = DataClass('/scratch/jcd496/Aemulus/DATA/training/scaled_full.csv')    
    training_data_loader = DataLoader(training_data, batch_size=len(training_data), shuffle=True, num_workers=args.workers)
    print("training data size", len(training_data))
    
    branch_net = BranchNet()
    optimizer = optim.Rprop(branch_net.parameters())
    branch_net.to(device)
    
    #COMMENT BELOW TO TRAIN
    if(args.load_path):
        state = torch.load('sicn_18000.rprop.pt')
        branch_net.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
    

    #UNCOMMENT BELOW TO TRAIN 
    outputs, labels, training_loss = train(branch_net, optimizer, args.loss_func, training_data_loader, args.epochs, device)
    if(args.save_path):
        torch.save({'model': branch_net.state_dict(),'optimizer': optimizer.state_dict()}, args.save_path)

    test_data = DataClass('/scratch/jcd496/Aemulus/DATA/test/scaled_test_data.csv')
    test_data_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False, num_workers=args.workers)
    print("test data size", len(test_data))

    labels, output, test_loss, fractional_error = test(branch_net, args.loss_func, test_data_loader,len(test_data), device)
    print("Fractional error:", np.average(fractional_error))
    #print("predictions", output[-9:])
    """if args.loss_func == 'chi-squared':
        print("pointwise:", training_chisquares_pointwise.shape)
        print("avg:", np.average(training_chisquares_pointwise))
        print("variance", np.var(training_chisquares_pointwise))
        plt.yscale('log',basey=10)
        plt.plot(training_chisquares_pointwise,'o')
    #plt.plot(output,'^')
        plt.show()"""
    
