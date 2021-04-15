import numpy as np

def cubicsolver(coef):
    
    a = coef[0]
    b = coef[1]
    c = coef[2]
    d = coef[3]

    f = findF(a, b, c)                          # Helper Temporary Variable
    g = findG(a, b, c, d)                       # Helper Temporary Variable
    h = findH(g, f)                             # Helper Temporary Variable

    i = np.sqrt(((g ** 2.0) / 4.0) - h)   # Helper Temporary Variable
    j = i ** (1 / 3.0)                      # Helper Temporary Variable
    k = np.arccos(-(g / (2 * i)))           # Helper Temporary Variable
    L = j * -1                              # Helper Temporary Variable
    M = np.cos(k / 3.0)                   # Helper Temporary Variable
    N = np.sqrt(3) * np.sin(k / 3.0)    # Helper Temporary Variable
    P = (b / (3.0 * a)) * -1                # Helper Temporary Variable

    x1 = 2 * j * np.cos(k / 3.0) - (b / (3.0 * a))
    x2 = L * (M + N) + P
    x3 = L * (M - N) + P

    return np.array([x1, x2, x3])           # Returning Real Roots as numpy array.
# Helper function to return float value of f.
def findF(a, b, c):
    return ((3.0 * c / a) - ((b ** 2.0) / (a ** 2.0))) / 3.0


# Helper function to return float value of g.
def findG(a, b, c, d):
    return (((2.0 * (b ** 3.0)) / (a ** 3.0)) - ((9.0 * b * c) / (a **2.0)) + (27.0 * d / a)) /27.0


# Helper function to return float value of h.
def findH(g, f):
    return ((g ** 2.0) / 4.0 + (f ** 3.0) / 27.0)


def calcFundamentalStats(y):
    ass = np.sum(y[1:-1]**2,axis=0)
    aep = y[0]**2 + y[-1]**2
    ac = np.sum(y[0:-1]*y[1:],axis=0)
    return aep,ass,ac

def calcBfromDataN(aep,ass,ac,N):
    coef = np.array([(N-1)*ass,
       (2.0-N)*ac,
       -aep-(N+1)*ass,
       N*ac])
    b = cubicsolver(coef)
    return b[2,:,:]

def calcAfromB(B,aep,ass,ac,N):
    Q=aep/(1-B**2)
    Q=Q+ass*(1+B**2)/(1-B**2)
    Q=Q-ac*2*B/(1-B**2)
    A = Q/N
    P2A = -N/A**2/2
    Btmp = B**2*(1+2*N)
    tmp = (1+Btmp)*aep + (2*Btmp + N + 1 -B**4*(N-1))*ass - 2*B*(1+B**2+2*N)*ac
    P2B = -tmp/((1-B**2)**2*(aep + (1+B**2)*ass - 2*B*ac))
    PAB = (N-1)*B/A/(1-B**2)
    dA = np.sqrt(-P2B/(P2A*P2B-PAB**2))
    dB = np.sqrt(-P2A/(P2A*P2B-PAB**2))
    return A,dA,dB

def calcCorr(windows,N,REGIONS):
    corrC = []
    corrdC = []
    corrB1 = []
    for m in windows:
        x1 = np.repeat(m[:, :, np.newaxis], REGIONS, axis=2)
        x2 = np.repeat(m[:, :, np.newaxis], REGIONS, axis=2).swapaxes(1,2)
        y1 = x1 + x2
        y2 = x1 - x2
        aep1,ass1,ac1 = calcFundamentalStats(y1)
        aep2,ass2,ac2 = calcFundamentalStats(y2)
        B1 = calcBfromDataN(aep1,ass1,ac1,N)
        B2 = calcBfromDataN(aep2,ass2,ac2,N)
        A1,dA1,dB1 = calcAfromB(B1,aep1,ass1,ac1,N)
        A2,dA2,dB2 = calcAfromB(B2,aep2,ass2,ac2,N)
        Adiff = A1-A2
        C = np.where(Adiff>0,Adiff/A2,Adiff/A1)
        np.fill_diagonal(C, 1.0)
        dC = np.where(Adiff>0,np.sqrt(dA1**2/A1**2 + A1**2*dA2**2/A2**4),np.sqrt(dA2**2/A1**2 + A2**2*dA1**2/A1**4))
        np.fill_diagonal(dC, 0.0)
        corrC.append(C)
        corrdC.append(dC)
        corrB1.append(B1)
    return corrC,corrdC,corrB1
