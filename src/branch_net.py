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
from sklearn.tree import DecisionTreeClassifier
HOD=8
LABEL=10+7
ERROR = 11+7
NPL = 128  #neurons per layer
CNPL= 4048
#data class.  loads data from csv file which is stored column wise.  column 0-7 hod training params, column 8 radius training param,
#column 9 regression target, column 10 error on data point
class DataClass(Dataset):

    def __init__(self, csv_file):
        self.file = np.genfromtxt(csv_file, delimiter=",")
        self.labels = self.file[:,16]
        self.hod_params = self.file[:,7:15]
        self.radius = self.file[:,15:16]
        self.error = self.file[:,17]
        self.cosmo_params = self.file[:,:7]
        

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        hod_param = self.hod_params[idx]
        labels = self.labels[idx]
        radii = self.radius[idx]
        error = self.error[idx]
        cosmo_param = self.cosmo_params[idx]
        sample = {'inputs' : {'cosmo_params': cosmo_param, 'hod_params': hod_param, 'radius': radii} , 'label': labels, 'error':error}
        return sample

#branch neural net.  causes radius parameter to "feel" more effect of loss function and reduces computation
class BranchNet(nn.Module):
    def __init__(self):
        super(BranchNet, self).__init__()
        #self.cosmo_fc1 = nn.Linear(COSMO,7)
        #self.cosmo_fc2 = nn.Linear(7,1)
        	
        self.hod_fc1 = nn.Linear(8,NPL)
        self.hod_fc2 = nn.Linear(NPL,NPL)
        self.hod_fc3 = nn.Linear(NPL,NPL)  #change 7 to 256 in each of above
        self.hod_fc4 = nn.Linear(NPL,1)  #change 7 to 256 in each of above
        
        self.combo_fc1 = nn.Linear(2, CNPL*3) #3,256
        self.combo_fc2 = nn.Linear(CNPL*3,CNPL*3)  #256,1
        self.combo_fc3 = nn.Linear(CNPL*3,1)  #256,1
        #self.combo_fc4 = nn.Linear(128, 1)


    def forward(self, x):
        """cosmo_out = self.cosmo_fc1(x['cosmo_params'].float())
        cosmo_out = torch.tanh(cosmo_out)
        cosmo_out = self.cosmo_fc2(cosmo_out)"""


        hod_out = self.hod_fc1(x['hod_params'].float())
        hod_out = torch.tanh(hod_out)
        hod_out = self.hod_fc2(hod_out)
        hod_out = torch.tanh(hod_out)
        hod_out = self.hod_fc3(hod_out)
        hod_out = torch.tanh(hod_out)
        hod_out = self.hod_fc4(hod_out)
        

        combo_in = torch.cat((hod_out, x['radius'].float()), 1)
        out = self.combo_fc1(combo_in)
        out = torch.tanh(out)
        out = self.combo_fc2(out)
        out = torch.tanh(out)
        out = self.combo_fc3(out)
        #out = torch.tanh(out)
        #out = self.combo_fc4(out)
        return out

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

"""class ChiSquare(Function):
    @staticmethod
    def forward(ctx, predicted, actual, error):
        diff = torch.add(predicted,-1,actual)
        power=torch.pow(diff,2)
        error_sq=torch.pow(error,2)
        ctx.save_for_backward(diff, error_sq)
        ratio = torch.div(power,error_sq)
        result = torch.sum(ratio)/len(predicted)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        diff, error_sq = ctx.saved_tensors
        grad_predicted = grad_actual = grad_error = None
        grad_predicted = 2.0*torch.div(diff,error_sq)
        grad_predicted = torch.neg(grad_predicted)
        return grad_predicted, grad_actual, grad_error"""

# train off of chisquare loss function to encapsulate error on training data
def ChiSquare(outputs, labels, errors):
    diff = torch.add(outputs,-1,labels)
    power=torch.pow(diff,2)
    error_sq=torch.pow(errors,2)
    ratio = torch.div(power,error_sq)
    loss = torch.sum(ratio)/len(outputs)
    return loss, ratio

def adjust_learning_rate(optimizer, epoch, rate):
    lr = rate*(0.5**(epoch//200))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#train network and keep statistics
def train(model, optimizer, loss_func, data_loader, epochs, scheduler, laps):
    start_time = monotonic()
    if loss_func=='chi-squared':
        criterion = ChiSquare
    else:
        criterion = nn.MSELoss()
    epoch_time = timer()
    load_time = timer()
    compute_time = timer()
    running_loss=0.0
    outputs =None
    chi_square_per_epoch=[]
    chi_square_per_point=[]
    for epoch in range(epochs):
        epoch_time_s = monotonic()
        running_loss = 0.0
        load_time_s = monotonic()
        chi_square_per_point=[]		
        for i, data in enumerate(data_loader, 0):
        	#load and keep stats for c2 and c3
        	inputs = data['inputs']
        	labels = data['label']
        	errors = data['error']

        	errors = errors.float()
        	labels=labels.float()
        	#inputs=inputs.view(-1,1)
        	labels = labels.view(-1,1)
        	errors = errors.view(-1,1)
        	#inputs, labels = inputs.to(device), labels.to(device)
        	load_time_f = monotonic()
        	

        	#feed forward, calc gradients, update params and keep stats for c2
        	comp_time_s = monotonic()			
        	optimizer.zero_grad()
        	outputs = model(inputs)

        	if loss_func == 'chi-squared':
        		pred_chi = criterion(outputs, labels, errors)
        		loss=pred_chi[0]
        		chi_square_per_point=np.append(chi_square_per_point,pred_chi[1].detach().numpy())
        	else:
        		loss = criterion(outputs, labels)
        	
        	loss.backward()
        	optimizer.step()	
        	comp_time_f = monotonic()
        	
        	#stats, running loss
        	load_time.update((load_time_f-load_time_s))
        	compute_time.update((comp_time_f-comp_time_s))
        	running_loss+=loss.item()

        chi_square_per_epoch.append(running_loss)
        epoch_time_f = monotonic()
        epoch_time.update((epoch_time_f-epoch_time_s))
        #scheduler.step(running_loss)
        #adjust_learning_rate(optimizer, epoch+100*laps, 0.001)
        print("epoch:", epoch+100*laps, "chi-square:", np.average(chi_square_per_point))
    end_time = monotonic()
    labels = labels.detach().numpy()
    outputs = outputs.detach().numpy()
    print("Total Time: {:.3f}    Num Epochs: {:d}   Average Epoch: {:.3f}   Average Load: {:.3f}    Average FWD+BWD: {:.3f}    Final Loss: {:.3f}".format(end_time-start_time, epochs, epoch_time.average(),load_time.average(), compute_time.average(), running_loss))
    return outputs, labels, chi_square_per_epoch, chi_square_per_point

#test network and report statistics
def test(model, loss_func, data_loader, size):
    test_loss = 0.0
    Labels = None
    output = None
    if loss_func=='chi-squared':
        criterion = ChiSquare
    else:
        criterion = nn.MSELoss().to(device)
    fractional_error=np.zeros((size,1),dtype=float) 
    for i, data in enumerate(data_loader, 0):
        	inputs = data['inputs']
        	labels = data['label']
        	errors = data['error']
        	errors=errors.float()
        	errors=errors.view(-1,1)
        	labels=labels.float()
        	labels = labels.view(-1,1)
        	#model.eval()
        	output = model(inputs)

        	if loss_func == 'chi-squared':
        		test_loss+=criterion(output,labels, errors)[0]
        	else:
        		test_loss+=criterion(output, labels)
        	print("test chi-squared:",test_loss)
        	fractional_error+=abs(output.detach().numpy()-labels.detach().numpy())/labels.detach().numpy()*100
    labels = labels.detach().numpy()
    output = output.detach().numpy()
    return labels, output, test_loss, fractional_error



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='emulator net')
    parser.add_argument('--data_path', type=str, default='../DATA/training/training_data_complete.csv',help='Data path')
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

    
    training_data = DataClass('/home/jcd496/aemulus/DATA/training/cosmo_breakdown/cosmo_0_training_data.csv')    
    training_data_loader = DataLoader(training_data, batch_size=2000, shuffle=True, num_workers=args.workers)
    print("training data size", len(training_data))
    
    branch_net = BranchNet()
    #branch_net.to(device)

    if args.loss_func is 'chi-squared':
        lrate = 0.001
    else:
        #criterion = nn.MSELoss()
        lrate = 0.00001
    optimizer = optim.SGD(branch_net.parameters(), lr=lrate, momentum=0.9,nesterov=False) 
    #optimizer = optim.Adam(branch_net.parameters()) 
    print("learning rate: ", lrate)
    #state = torch.load('model.pt')
    #branch_net.load_state_dict(state['model'])
    #optimizer.load_state_dict(state['optimizer'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    #UNCOMMENT BELOW TO TRAIN 
    chisq=500
    laps=0
    while chisq>200:
        outputs, labels, training_chisquares, training_chisquares_pointwise = train(branch_net, optimizer, args.loss_func, training_data_loader, args.epochs, scheduler, laps)
        chisq=np.average(training_chisquares_pointwise)
        print("training chi-square:", chisq )
        laps+=1

    test_data = DataClass('/home/jcd496/aemulus/DATA/training/cosmo_breakdown/cosmo_0_test_data.csv')
    test_data_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=True, num_workers=args.workers)
    print("test data size", len(test_data))
    #COMMENT BELOW TO TRAIN, model.1 PRETRAINED AFTER 2000 EPOCHS ON HPC, model.pt is 1000 epochs. APPROX 60 MINUTES
    #branch_net.load_state_dict(torch.load('model.pt'))

    labels, output, test_loss, fractional_error = test(branch_net, args.loss_func, test_data_loader,len(test_data))
    print("Fractional error:", np.average(fractional_error))
    torch.save({'model': branch_net.state_dict(), 'optimizer': optimizer.state_dict()} , 'model.pt')
    """if args.loss_func == 'chi-squared':
        print("pointwise:", training_chisquares_pointwise.shape)
        print("avg:", np.average(training_chisquares_pointwise))
        print("variance", np.var(training_chisquares_pointwise))
        plt.yscale('log',basey=10)
        plt.plot(training_chisquares_pointwise,'o')
    #plt.plot(output,'^')
        plt.show()"""
    
