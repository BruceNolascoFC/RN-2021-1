import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
import numpy as np
from torch.autograd.functional import jacobian
import crane
from crane import C,LMAX,LMIN,LVMAX,XMAX,UMAX

class Dynamic(torch.nn.Module):
    def __init__(self):
      super(Dynamic, self).__init__()
      # Layers of the NN
      self.layer1 =nn.Sequential( nn.Linear(8,10), nn.ReLU6(),
      nn.Linear(10,10), nn.PReLU(),
      nn.Linear(10,10), nn.Tanh(),
      nn.Linear(10,10), nn.Tanh() )

      self.layer2 =nn.Bilinear(4,4,4)
      self.layer3 =nn.Sequential( nn.Linear(6,6), nn.Tanh())

      self.layer4 =nn.Sequential( nn.Linear(10,8), nn.Tanh()
      ,nn.Linear(8,8), nn.Tanh()
      ,nn.Linear(8,8), nn.Tanh()
      ,nn.Linear(8,6), nn.Tanh())

      self.us = None
      self.ls = None
    def forward(self,t, input):
        # Note: first entry of input is time
        signal_tensor = torch.tensor([self.us(t),self.ls(t)])
        a1 = self.layer1(torch.hstack([signal_tensor,input]).float())
        a2 = self.layer2(a1[:4],a1[:4])
        a3 = self.layer3(a1[4:])
        output = self.layer4(torch.hstack([a2,a3]))
        # Transform each entry to a suitable magnitude
        s = torch.zeros(6)
        s[0] = output[0]
        s[1] = output[1]
        s[2] = LVMAX*output[2]
        s[3] = 10*output[3]
        s[4] = UMAX*output[4]
        s[5] = 10*output[5]

        return s
    def set_signal(self, us,ls):
        self.us = us
        self.ls = ls
    def train_from_case(self,case,alpha = 0.001,method = "BDF"):
        # We set the signal for the forward pass
        self.set_signal(case["us"],case["ls"])
        t0 = case["t0"]
        tf = case["tf"]
        y0 = case["y0"].flatten()
        yf = case["yf"].flatten()
        statesize = y0.shape
        statelen = np.prod(statesize)
        # numpy compatible function
        df = integrable_from_model(self,statesize)
        # Solve the ODE forward in time
        out = solve_ivp(df,(t0,tf),y0,method = method,t_eval = [tf])
        # Returns to Tensor form
        h = torch.tensor(out["y"].flatten(),requires_grad = True)
        yft = torch.tensor(yf,requires_grad= True)
        # we compute the loss with MSE criterion
        L = torch.nn.MSELoss()(yft,h) /(tf-t0)
        L.backward()
        grad_output = h.grad

        # first we flatten grad_output as it is $dL/dz(t_f)$
        atf = grad_output.numpy().flatten()
        adjshape = atf.shape
        # length of adjoint state
        adjlen = np.prod(atf.shape)
        # we clone the model to avoid detachment from main autograd graph
        copy= Dynamic() # get a new instance
        copy.load_state_dict(self.state_dict())
        copy.set_signal(case["us"],case["ls"])
        # length of gradient wrt parameters
        thlens = []
        thshapes = []
        thetagradlen = 0
        for param in copy.parameters():
            temp = torch.clone(param).detach().numpy()
            thetagradlen += np.prod(temp.shape)
            thlens.append(np.prod(temp.shape))
            thshapes.append(temp.shape)
        # initial augmented state
        s0 = np.hstack([out["y"].flatten(),atf,np.zeros(thetagradlen)])
        # function that computes the augmented
        # state dynamics
        def aug_dynamics(t,s):
            # unpack current state
            z = s[0:statelen]
            # compute dz/dt
            zdot = df(t,z)
            # unpack adjoint state
            a = s[statelen:statelen+adjlen]
            adj = np.reshape(a,adjshape)
            # da/dt using Pontryagin method
            adjdot= -adj@input_jacobian(copy,t,z)
            # d(dL/dtheta)/dt using the augmented state method
            dthetadot= -adj@flat_param_jacobian(copy,t,z)
            return np.hstack([zdot,adjdot,dthetadot])
        # We solve back in time the augmented dynamics
        jacobian_vector = solve_ivp(aug_dynamics,(tf,t0),s0,method = method,t_eval = [t0])
        jacobian_vector = jacobian_vector["y"].flatten()
        # extract just parameter jacobian
        jacobian_vector = jacobian_vector[statelen+adjlen:]
        # we update the models parameters
        vec = nn.utils.parameters_to_vector(self.parameters()).detach().numpy()
        vec -= alpha* jacobian_vector
        nn.utils.vector_to_parameters(torch.tensor(vec),self.parameters())
        return L,yft,h
    def ode_solver(self,y0,t0,tf,signals = None,dense = False):
        if signals != None:
            auxsignal = (self.us,self.ls)
            self.set_signal(signals[0],signals[1])
        statesize = y0.shape
        # numpy compatible function
        df = integrable_from_model(self,statesize)
        # Solve the ODE
        out = solve_ivp(df,(t0,tf),y0,method = method,t_eval = [tf],dense_output=dense)
        # Returns to Tensor form
        h = torch.tensor(out["y"].flatten(),requires_grad = True)

        if signals != None:
            self.set_signal(auxsignal[0],auxsignal[1])

        if dense:
            return h,out.sol
        else:
            return h
    def train_rounds(n,path = "",alpha=0.001,method= "BDF"):
         losses = []
         save = len(path)>0
         for i in range(n):
            case = build_case()
            print("Case built")
            loss,y,h = model.train_from_case(case,alpha=0.0005,method= method)
            print("Training donde EPOCH ",i," ",loss.item()/(case["tf"]-case["t0"])," ",(case["tf"]-case["t0"]))
            print(y.data)
            print(h.data)
            losses.append(loss/(case["tf"]-case["t0"]))
            if (i%10 == 0) and save:
                torch.save(model.state_dict(), path)
                print("MODEL SAVED")


def x(state):
    return state[4]+state[2]*torch.sin(state[0])
def y(state):
    return -state[2]*torch.cos(state[0])
def cartesian_criterion(state1,state2):
    return torch.sqrt(torch.sum((x(state1)-x(state2))**2+(y(state1)-y(state2))**2))

def input_jacobian(model,t,z,numpy_comp = True):
    """
    Uses autograd to return a function that returns the
    tensor jacobian of the models output with respect to the
    input z.
    @param NN Instance model
    @param z Input (pref depth = 1)
    @param numpy_comp If true, returns numpy array.
        Otherwise returns Torch tensor.
    """
    aux = lambda x: model.forward(t,x)
    J = jacobian(aux,torch.tensor(z))
    if numpy_comp:
        J = torch.clone(J).detach().numpy()
    return J

def flat_param_jacobian(model,t,z,numpy_comp = True):
    """
    Returns the jacobian of the model output with z as
    input with respect to the flattened parameter vector.
    Has dimensions model.forward x num. params.
    z must have depth of 1.

    @param numpy_comp If true, returns numpy array.
        Otherwise returns Torch tensor.
    """
    copy=type(model)() # get a new instance
    copy.load_state_dict(model.state_dict()) # copy parameters
    copy.set_signal(model.us,model.ls)
    out = copy.forward(t,torch.tensor(z)) # feed forward pass
    rows = [] # list of the jacobian rows
    for i in range(len(out)):
        # backpropagation of output i-th component
        #  retains graph as trick to mantain O(1) memory
        out[i].backward(retain_graph= True)
        row = []
        for param in copy.parameters():
            row.append(param.grad.numpy().flatten())
        rows.append(np.hstack(row))
    # stack all jacobian rows
    J = torch.tensor(np.vstack(rows))
    if numpy_comp:
        J = torch.clone(J).detach().numpy()
    return J

def integrable_from_model(model,input_shape):
    """
    Given an Dynamic model and input_shape.
    Returns an integrable function that accepts and returns flatten numpy
    arrays recoverable to model's input shape.
    """
    def f(t, input, options = None):
        input = torch.reshape(torch.from_numpy(input), input_shape)
        z = model.forward(t,input)
        z = z.detach().numpy().flatten()
        return z
    return f
