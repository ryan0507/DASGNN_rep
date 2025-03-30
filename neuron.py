from typing import List, Tuple
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

gamma = 0.2
thresh_decay = 0.7

def reset_net(net: nn.Module):
    for m in net.modules():
        # Make reset all of the neuron with thresholod values 
        if hasattr(m, 'reset'):
            m.reset()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Spiking Functions with surrogate functions
'''
class BaseSpike(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x, alpha)
        return x.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError


class SuperSpike(BaseSpike):
    """
    Spike function with SuperSpike surrogate gradient from
    "SuperSpike: Supervised Learning in Multilayer Spiking Neural Networks", Zenke et al. 2018.

    Design choices:
    - Height of 1 ("The Remarkable Robustness of Surrogate Gradient...", Zenke et al. 2021)
    - alpha scaled by 10 ("Training Deep Spiking Neural Networks", Ledinauskas et al. 2020)
    """

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        sg = 1 / (1 + alpha * x.abs()) ** 2
        return grad_input * sg, None


class TriangleSpike(BaseSpike):
    """
    Spike function with triangular surrogate gradient
    as in Bellec et al. 2020.
    """
    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        sg = torch.nn.functional.relu(1 - alpha * x.abs())
        return grad_input * sg, None


class ArctanSpike(BaseSpike):
    """
    Spike function with derivative of arctan surrogate gradient.
    Featured in Fang et al. 2020/2021.
    """
    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        sg = 1 / (1 + alpha * x * x)
        return grad_input * sg, None


class SigmoidSpike(BaseSpike):
    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        grad_input = grad_output.clone()
        sgax = (x * alpha).sigmoid_()
        sg = (1. - sgax) * sgax * alpha
        return grad_input * sg, None

# Surrogate function
def superspike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(10.0)):
    return SuperSpike.apply(x - thresh, alpha)

def sigmoidspike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(1.0)):
    if x.device != thresh.device:
        thresh = thresh.to(x.device)
    if x.device != alpha.device:
        alpha = alpha.to(x.device)
    return SigmoidSpike.apply(x - thresh, alpha)

def trianglespike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(1.0)):
    return TriangleSpike.apply(x - thresh, alpha)

def arctanspike(x, thresh=torch.tensor(1.0), alpha=torch.tensor(10.0)):
    return ArctanSpike.apply(x - thresh, alpha)

# Surrogation with dictionary
SURROGATE = {'sigmoid': sigmoidspike, 
             'triangle': trianglespike, 
             'arctan': arctanspike,
             'super': superspike}


        
class AALIF(nn.Module):
    def __init__(self,size,  tau=1.0, v_threshold=1.0, v_reset=0., alpha=1.0, surrogate='triangle'):
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.surrogate = SURROGATE.get(surrogate)
        self.register_buffer("tau", torch.as_tensor(tau, dtype=torch.float32))
        self.register_buffer("alpha", torch.as_tensor(
            alpha, dtype=torch.float32))
        self.reset()
        self.v_threshold_values = []  # List to store v_threshold values during forward passes
        

    def reset(self):
        self.v = 0.
        self.v_th = self.v_threshold

    def forward(self, dv):
        # 1. charge
        self.v = self.v + (dv - (self.v - self.v_reset)) / self.tau
        # 2. fire
        spike = self.surrogate(self.v, self.v_th, self.alpha)
        # 3. reset
        self.v = (1 - spike) * self.v + spike * self.v_reset
        # 4. threhold updates
        # Calculate change in cell's threshold based on a fixed decay factor and incoming spikes.
        self.v_th = gamma * spike + self.v_th * thresh_decay
        self.v_th = torch.mean(self.v_th, axis = 0)
        # print(self.v_th.size())
        
        if self.training:
            if torch.is_tensor(self.v_th):
                mean_val = torch.mean(self.v_th)            
                self.v_threshold_values.append(mean_val.item())
                # print(self.v_th.size())
                indices = torch.nonzero(spike == 1.0)
                # print(self.v_th[indices])
        return spike

    def are_all_v_th_same(self):
        if torch.is_tensor(self.v_th) and len(self.v_threshold_values) > 1:
            # Check if all values in v_threshold_values are close to the first value
            return torch.allclose(self.v_th, torch.Tensor(self.v_threshold_values[0]))
        else:
            # If there's only one value or it's not a tensor, return True
            return True

########### DEGREE + FEATURE NEURON ############ 

class BaseNeuron_degree_feat(nn.Module):
    def __init__(self,ssize=128, tau: float =1.0, v_threshold: float=0.25, v_reset: float=0., alpha: float=1.0, 
                 surrogate: str = 'triangle', threshold_trainable : bool = False, init_multi = False):
        
        '''
        tau (float): dacay values for v_tthreshold
        v_thresehold (float): Threshold whether omit spikes or not
        v_reset (float): reet values could be adjusted
        alpha (float): Smoothing Factor for surrogate function
        surrogate (float): Surrogate Functions [simoid, triangle, arctan, super]
        '''
        
        super().__init__()
        self.v_reset = v_reset
        self.v = 0.
        self.train_spike_counts = None
        self.valid_spike_counts = None
        self.train_cur_degree = None
        self.valid_cur_degree = None
        
        # if threshold_trainable:
        # linear_spaced_tensor = torch.linspace(0.0, 30.0, steps=ssize, dtype=torch.float32)
        #     self.register_parameter("v_threshold", nn.Parameter(
        #         torch.as_tensor(linear_spaced_tensor, dtype=torch.float32)
        #     ))
            
        try:
            self.surrogate = SURROGATE[surrogate]
        except:
            print('Unvailable surrogate function. Please check surrogate functions')
         
        
        # Tau and alpha (smoothing factor) should not be updated
        
        # Check v_threshold values for trainable (default: False)
        if isinstance(self, LAPLIF_deg_feat):
            self.register_parameter("tau", nn.Parameter(
                torch.as_tensor(tau, dtype=torch.float32)))
            print('Tau paramter')
        else:
            self.register_buffer("tau", torch.as_tensor(tau, dtype=torch.float32))
            print('Tau buffer')
        
        self.register_buffer("alpha", torch.as_tensor(
            alpha, dtype=torch.float32))
        
        self.v_reset_channel = 0.
        # Reset to Initial input values
        # self.reset() 
        

    def reset(self):
        '''
        Reset all of the Neuron states
        self.v : Tensor[size] Self neuron state that all of the neuron 
        implicitly own itself.
        self.v_th : Set threshold to omit spikes, it cloud to adjust for threshold values
        '''
        self.v = 0.
        
        # if not isinstance(self, LIFboth):
        # self.v_th = self.v_threshold
        
    def calibrated_neuron(self):
        eps = 1e-7
        if type(self.v) is float:
            max_v = 0
            min_v = 0
        else:
            max_v = torch.max(self.v)
            min_v = torch.min(self.v)
            # print(max_v, min_v)
            # print(self.v[self.v > 0].all())
            
        self.v = (self.v - min_v) / (max_v - min_v + eps)
           
    def forward(self, dv):
        raise NotImplementedError
    
    def save_neuron_spikes(self, path):
        torch.save(self.train_spike_counts, f"{path}_train_rate.pt")
        torch.save(self.train_cur_degree, f"{path}_train_cur_degree.pt")
        torch.save(self.valid_spike_counts, f"{path}_valid_rate.pt")
        torch.save(self.valid_cur_degree, f"{path}_valid_cur_degree.pt")
    
    def update_spike_counts(self, degree, cur_spike):
        
        if self.training:
            self.train_cur_degree = degree
            if self.train_spike_counts is None:
                self.train_spike_counts = cur_spike
            else:
                self.train_spike_counts += cur_spike
        else:
            self.valid_cur_degree = degree
            if self.valid_spike_counts is None:
                self.valid_spike_counts = cur_spike
            else:
                self.valid_spike_counts += cur_spike
            
    
    def reset_stat(self):
        self.train_spike_counts = None
        self.valid_spike_counts = None
        
class LIF_deg_feat(BaseNeuron_degree_feat):
    '''
    Leaky Integrated Fire models (LIF type)
    '''
    
    def __init__(self, ssize=128, tau=1.0, v_threshold=0.25, v_reset=0.0, alpha=1.0, 
                 surrogate='sigmoid', threshold_trainable : bool = False, bins = 20):
        super().__init__(ssize, tau, v_threshold, v_reset, alpha, surrogate, threshold_trainable)
        
        # List to store v_threshold values during forward passes
        self.bins = bins
        self.v_threshold_values = []
        # self.spike_counts = None
        # self.cur_threshold = None
        self.init_threshold = v_threshold
        initial_tensor = torch.as_tensor([v_threshold] * bins, dtype=torch.float32)
        v_threshold_tensor = initial_tensor.unsqueeze(1).expand(bins, ssize)
        
        if threshold_trainable:
                self.register_parameter("v_threshold", nn.Parameter(
                v_threshold_tensor.clone()
            ))
        else:
            self.register_buffer("v_threshold", nn.Parameter(
                v_threshold_tensor.clone()
            ))
        
    def forward(self, dv, degree, orig_degree = None):
        '''
        dv (Tensor) : input size and output size automatically given
        '''
        self.cur_degree = orig_degree
        self.v = self.v + (dv - (self.v - self.v_reset)) / self.tau
        # Surrogated -> v_th should not be changed for this neuron
        spike = torch.zeros_like(self.v)
        total_degree = torch.unique(degree).tolist()
        for cur_degree in total_degree:
            spike[degree == cur_degree] = self.surrogate(self.v[degree == cur_degree], self.v_threshold[cur_degree], self.alpha)
        self.v = (1 - spike) * self.v + spike * self.v_reset
        
        if self.training:
            if torch.is_tensor(self.v_threshold):
                # self.cur_threshold = self.v_threshold
                # print(self.v_threshold)
                mean_val = torch.mean(self.v_threshold)            
                self.v_threshold_values.append(mean_val.item())
        self.update_spike_counts(orig_degree, spike)
        
        return spike

class Deg_feat_neuron(BaseNeuron_degree_feat):
    def __init__(self, ssize=128, tau=1.0, v_threshold=0.25, v_reset=0.0, alpha=1.0, 
                 surrogate='sigmoid', threshold_trainable : bool = False, bins=20):
        super().__init__(ssize, tau, v_threshold, v_reset, alpha, surrogate, threshold_trainable, bins)
        
        self.init_threshold = v_threshold
        initial_tensor = torch.as_tensor([v_threshold] * bins, dtype=torch.float32)
        v_threshold_tensor = initial_tensor.unsqueeze(1).expand(bins, ssize)
        
        self.register_parameter("v_threshold", nn.Parameter(
            v_threshold_tensor.clone()
        ))
        self.v_threshold_values = []  
        self.reset()
        
    def forward(self, dv, binned_degree, orig_degree=None):
        self.v = self.v + (dv - (self.v - self.v_reset)) / self.tau
        spike = torch.zeros_like(self.v)
        total_degree = torch.unique(binned_degree).tolist()
        for cur_degree in total_degree:
            spike[binned_degree == cur_degree] = self.surrogate(self.v[binned_degree == cur_degree], self.v_threshold[cur_degree], self.alpha)
        self.v = (1 - spike) * self.v + spike * self.v_reset
        with torch.no_grad():
            v_th_new = self.v_th.clone()
            for i in range(self.v_threshold.size(0)):
                mask = (binned_degree == i)
                if mask.any():
                    v_th_new[i] = gamma * spike[mask].mean(axis=0) + self.v_th[i] * (1 - gamma)
                
        self.v_th = nn.Parameter(v_th_new)
        
        if self.training:
            if torch.is_tensor(self.v_th):
                mean_val = torch.mean(self.v_th)            
                self.v_threshold_values.append(mean_val.item())
            
        self.update_spike_counts(orig_degree, spike)
        
        return spike
    
    def reset(self):
        self.v = 0.
        self.v_th = nn.Parameter(self.v_threshold.clone()) 
