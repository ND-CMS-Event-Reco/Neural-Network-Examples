''' Utils for creating data + training 
'''
import numpy as np
import torch
import torch.nn as nn
import math
import copy


####################################
#  Data generation + manipulation  #
####################################
def batchify(data, target):
    num_batch = int(math.floor(data.shape[0] / 100.0))
    input_batches = data.chunk(num_batch)
    output_batches = target.chunk(num_batch)
    
    return num_batch, input_batches, output_batches

def generateSeqs(N, in_len, data_range = (-1, 1), operation = 'subtraction', 
    method = 'neighbors', SOS_token = np.pi, EOS_token = -np.pi):
    ''' 
        Args:
            N (int)
            in_len (int)
            data_range (tuple) 
            operation (str)
            method (str)
            
        Out
            data
            target
    '''
    
    # Pull random floats in range (data_range[0], data_range[1])
    data = (data_range[0] - data_range[1]) * torch.rand((N, in_len)) + data_range[1]
    
    target = torch.zeros((N, in_len - 1))
    
    if operation == 'subtraction':
        op_func = subtractTensors
    elif operation == 'multiplication':
        op_func = multiplyTensors
    elif operation == 'division':
        op_func = divideTensors
    elif operation == 'power':
        op_func = exponentiateTensors
    
    for i in np.arange(0, in_len - 1, 1):
        target[:, i] = op_func(data[:, i], data[:, i + 1])
        
    # Appent SOS and EOS token
    SOS_ = SOS_token*torch.ones((N, 1))
    EOS_ = EOS_token*torch.ones((N, 1))

    data = torch.cat((SOS_, data), axis=1)
    data = torch.cat((data, EOS_), axis=1)

    target = torch.cat((SOS_, target), axis=1)
    target = torch.cat((target, EOS_), axis=1)

    # Unsqueeze so each entry becomes its own dimension, needed for the nn.Linear embedding
    data = torch.unsqueeze(data, dim=2)
    target = torch.unsqueeze(target, dim=2)
    
    return data, target
     
def subtractTensors(ten1, ten2):
    return ten1 - ten2
    
def multiplyTensors(ten1, ten2):
    return torch.mul(ten1, ten2)
    
def divideTensors(ten1, ten2):
    return torch.mul(ten1, 1/ten2)

def exponentiateTensors(ten1, ten2):
    return torch.pow(ten1, ten2)
    
   


###############################
#       Training              #
###############################   

def train_epoch(model, opt, criterion, scheduler, device, data_batches, target_batches, num_batch, verbose = False):
    ''' Train transformer model
    
        Args:
            model (FloatTransformer)
            opt
            criterion
            device (str)
            data_batches (tuple of tensors)
            target_batches (tuple of tensors)
            num_batch (int)
    '''
    model.train()
    total_loss = 0
    
    for i in range(num_batch):
        data = data_batches[i].to(device)
        target = target_batches[i].to(device)
        
        # Take SOS to last token before EOS as target input
        target_in = target[:, :-1].to(device)
        
        # Take first input to EOS as target expected from transformer
        target_expected = target[:, 1:].to(device)
        
        # Create masks
        tgt_mask = model.get_tgt_mask(target_in.size(1)).to(device)
        src_mask = model.get_tgt_mask(data.size(1)).to(device)
        
        pred = model(data, target_in, src_mask, tgt_mask)
        
        if i == 0 and verbose:
            print('data', data)
            print('tgt in', target_in)
            print('pred', pred)
            print('expected', target_expected)
        
        loss = (criterion(pred, target_expected).type(torch.float))
        
        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.detach().item()
        
    scheduler.step()
    return total_loss
    
def train(model, n_epochs, opt, criterion, scheduler, device, data_batches, target_batches, num_batch):
    best_loss = 10000.0
    best_model = copy.deepcopy(model).to(device)
    loss_ = np.array([])
    epochs = np.array([])
    
    for i in range(n_epochs):
        loss = train_epoch(model, opt, criterion, scheduler, device, data_batches, target_batches, num_batch)
        loss_ = np.append(loss_, loss)
        epochs = np.append(epochs, i)
        
        if loss < best_loss:
            best_loss = loss
            best_model = copy.deepcopy(model).to(device)
            
        if i % 10 == 0:
            print(f'Epoch: {i}\nTotal Loss: {loss}')
            print(f'-----------------------------------')
            
    return best_model, loss_, epochs
    
if __name__ == "__main__":
    src, tgt = generateSeqs(10, 3, operation = 'multiplication')
    print(src[0])
    print(tgt[0])