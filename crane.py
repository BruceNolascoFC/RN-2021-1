import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numpy import sin, cos

def randomGaussianWalk(starttime = 0.0, endtime = 10.0,samples = 100,initial = 0.0,drift = 0.0, std = 1.0):
    """
    Returns
    A timesamples
    B randomWalk values at each A point.
    signal Function that gives B value to each t.
    """
    A = np.linspace(starttime, endtime, samples)
    B = np.zeros(A.shape)

    B[0] = initial
    # scale drift and std
    totaldrift = (A[1]-A[0])*drift
    totalstd = std/np.sqrt(samples/(endtime-starttime))
    for i in range(1,len(A)):
        B[i] = totaldrift + normal(0,totalstd)+B[i-1]
        if abs(B[i])>1:
            B[i] = B[i]/abs(B[i])
    # the function that returns the value of the signal
    # for a given time
    def signal(t):
        if starttime<= t<= endtime:
            index = int((t-starttime)/(A[1]-A[0]))
            return B[index]
        else:
            return 0
    return A,B,signal

# Constantes de la grua
# Constante de amortiguamiento
C = 10.
# Lngitud maxima
LMAX = 2.
# longitud mínima
LMIN = 0.5
# maxima velocidad de retracción
LVMAX = 0.5
# longitud del riel del carro
XMAX = 15.
# maxma velocidad del carro
UMAX = 4.

#Las entradas se definen por el orden:
# theta, theta' ,L, L' , x , u
# 0        1     2  3    4   5
def randomState():
    """
    Returns an array of shape (6,) with random values.
    """
    y = np.zeros((6))
    y[0] = normal(scale=0.1)
    y[1] = normal(scale=0.1)
    y[2] = normal(scale=0.2)+1
    y[3] = normal(scale=0.1)
    y[4] = normal(scale=0.2)+5
    y[5] = normal(scale=0.1)

    return y

def dynamic_from_signals(us,ls):
    """
    Returns an integrable function of the crane dynamics
    given the control signals us and ls.
    """
    def fun(t,y):
        """
        Given t and the actual state y returns the time derivative of y.
        References the global variables US and LS which represent the control signals.
        """
        # the signal guides acceleration
        accu = C*(us(t)*UMAX-y[5])
        accl = C*(ls(t)*LVMAX-y[3])
        # initialize the velocity vector
        v = np.zeros((6))
        # angular speed
        v[0] = y[1]
        # cable length speed
        v[2] = y[3]
        ldot = y[3]
        # cable length acceleration
        v[3] = accl
        # cart speed
        v[4] = y[5]
        # cart acceleration
        v[5] = accu

        # Condition that limits the cable length
        if LMIN > y[2]:
            #print("L min out")
            v[2] = max(0,v[2])
            ldot = 0.
        elif LMAX < y[2]:
            #print("L max out")
            v[2] = min(0,v[2])
            ldot = 0.
        # Condition that limits the cable length velocity
        if -LVMAX > y[3]:
            #print("Lv min out")
            v[3] = max(0,v[3])
            ldot = -LVMAX
        elif LVMAX < y[3]:
            #print("L max out")
            v[3] = min(0,v[3])
            ldot = LVMAX
        # Condition that limits the cart position
        if 0 > y[4]:
            #print("X min out")
            v[4] = max(0,v[4])
            accu = 0.
        elif XMAX < y[4]:
            #print(t," X max out")
            v[4] = min(0,v[4])
            accu = 0.
        # Condition that limits the cart velocity
        if -UMAX > y[5]:
            #print("u min out")
            v[5] = max(0,v[5])
            accu = v[5]
        elif UMAX < y[5]:
            #print("u max out")
            v[5] = min(0,v[5])
            accu = v[5]
        # angular acc
        v[1] = -(2.*ldot*y[1]+9.81*sin(y[0])+accu*cos(y[0]))/y[2]

        return v
    return fun

def build_case():
    """
    From random conditions returns a dic of an initial and final state,times,
    cart speed and length speed signals.
    """
    # Take random gaussian walk
    _,_,usignal = randomGaussianWalk(endtime = 10.,drift = 0.,std = 0.3,samples = 300)
    _,_,lsignal = randomGaussianWalk(endtime = 10.,drift = 0.,std = 0.3,samples = 300)
    # sample  t0 and t1
    ts = np.random.choice(np.linspace(0.,10.,300),2)
    ts.sort()
    # solve ode
    out =solve_ivp(dynamic_from_signals(usignal,lsignal),(0.,10.),randomState(),method = "BDF",t_eval = ts)
    # pack in dic
    outdic = {
        "t0" : ts[0],
        "y0" : out["y"][:,0],
        "tf" : ts[1],
        "yf" : out["y"][:,1],
        "us" : usignal,
        "ls" : lsignal,
        }
    return outdic
