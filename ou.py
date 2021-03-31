# function to calculate A and B from the dataset
def calc_fundstats(x):
    return x[0]**2+x[-1]**2,np.sum(x[1:-1]**2),np.sum(x[0:-1]*x[1:])

def OUanalytic(aep,ass,ac,N):
    coef = [(N-1)*ass,
       (2.0-N)*ac,
       -aep-(N+1)*ass,
       N*ac]
    B=np.roots(coef)[-1]
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
    return A,dA,B,dB

def OUresult(aep,ass,ac,N,delta_t):
    A, dA, B ,dB = OUanalytic(aep,ass,ac,N)
    tau = -delta_t/np.log(B)
    dtau = delta_t*dB/B/np.log(B)**2
    return A,dA,tau,dtau

def OUcross(data1,data2):
    N = len(data1)
    x1 = data1 + data2
    x2 = data1 - data2
    aep1,ass1,ac1 = calc_fundstats(x1,N)
    aep2,ass2,ac2 = calc_fundstats(x2,N)
    x1_A,x1_dA,_,_= OUanalytic(aep1,ass1,ac1,N)
    x2_A, x2_dA,_,_= OUanalytic(aep2,ass2,ac2,N)
    if x1_A > x2_A:
        C = (x1_A - x2_A)/x2_A
        dC = np.sqrt(x1_dA**2/x1_A**2 + x1_A**2*x2_dA**2/x2_A**4)
    else:
        C = (x1_A - x2_A)/x1_A
        dC = np.sqrt(x2_dA**2/x1_A**2 + x2_A**2*x1_dA**2/x1_A**4)
    return C,dC
