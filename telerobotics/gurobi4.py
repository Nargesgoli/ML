# import gurobipy
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt

# Define constants
B = 180e-3
sigma_dBm = -173.9
sigma_squared = (10**((sigma_dBm - 30) / 10.0))  # Convert dBm to watts
# sigma_squared=-173.9
k = 50 # #number of PRBs
p_ij_dBm= -20
p_ij= (10**((p_ij_dBm-30) / 10.0))
# p_ij =1e-5
# p_ij =-0.2
# rho_ij = {}
D=1e-2
Ne=3
Nr=3
M=1
F_i0=[0]*Nr
X_s=0.95
time_slot_duration = 1  # 100 milliseconds
D_max=10e-3
# D_max=time_slot_duration
Cs=100
ai=1
average_packet_size = 10e-3
# Simulation parameters
num_time_slots = 100
time_slot_duration = 1  # 100 milliseconds
def generate_h_ij1():
    lambda_parameter = 2
    num_samples = 1000
    pi = np.pi
    alpha_values = (np.random.rand(num_samples) - 0.5) * 2 * pi
    phi_values = (np.random.rand(num_samples) - 0.5) * 2 * pi
    # Generate Rayleigh fading samples
    x1 = np.random.randn(num_samples) * np.cos(phi_values)
    y1 = np.random.randn(num_samples) * np.sin(phi_values)
    # Calculate complex channel response
    z1 = (1 / np.sqrt(num_samples)) * (x1 + 1j * y1)
    # Calculate gain (magnitude of the channel response)
    gain = np.abs(z1)
    h_ij_random = np.random.choice(gain)
    return h_ij_random   
# Generate Rayleigh gain samples
# h_ij=generate_h_ij()
def generate_h_ij():
    lambda_parameter = 2
    path_loss_exponent = 3
    carrier_frequency = 900e6  # 900 MHz
    transmitted_power = 20  # Example transmitted power in dBm
    cell_radius = 1.5e3  # Cell radius in meters
    edge_distance = cell_radius
    path_loss_dB = -10*path_loss_exponent * np.log10(edge_distance)
    num_samples = 1000
    rayleigh_fading_samples = np.sqrt(np.random.exponential(scale=0.5, size=num_samples)) * np.exp(1j * 2 * np.pi * np.random.rand(num_samples))
    channel_gain=np.abs(rayleigh_fading_samples)
    gain_random=np.random.choice(channel_gain)
    return gain_random

def generate_delta_x():
    mu, sigma = 0.5, 0.2  # Mean and standard deviation
    delta_x_normal = np.random.normal(mu, sigma, Nr)
    delta_x_normal_binary = np.clip(delta_x_normal, 0, 1)
    delta_x_random=np.random.choice(delta_x_normal_binary,size=Nr)
    # print("delta_x=",delta_x) 
    return delta_x_random+0.001 
# delta_x=generate_delta_x() 

# rho_ij = np.random.choice([0, 1], size=(10, 4), p=[0.5, 0.5])
# r_tau=np.zeros((Ne+M,1))
m = gp.Model()
# lam= m.addVar(name=f'lambda',vtype=GRB.CONTINUOUS)
def generate_rho_ij(sum,k):
    rho_ij={}
    for i in range(sum):
        for j in range(k):
            rho_ij[i, j] = m.addVar(name=f'rho_{i}_{j}', vtype=GRB.BINARY)
    return rho_ij
# def generate_rho_ij(sum,k):
#     sum=Ne+M+Nr
#     rho_ij={}
#     for i in range(sum):
#         if i < Ne+M:
#             for j in range(k):
#                 rho_ij[i, j] = m.addVar(name=f'rho_{i}_{j}', vtype=GRB.BINARY)
#         else:
#             for j in range(k):
#                 rho_ij[i, j] = m.addVar(name=f'rho_{i}_{j}', vtype=GRB.BINARY)           
#     return rho_ij
def generate_r_tau( sum,k,p_ij,sigma_squared,B,h_ij,rho_ij):
    r_tau=list([0]*(Ne+M+Nr))
    s=np.array(0)
    for i in range(sum):
        # h_ij=generate_h_ij()
        for j in range(k ):
            s=s+(B * (rho_ij[i, j] * (np.log2(1 + (p_ij *h_ij) / (sigma_squared)))))
        r_tau[i]=s
        s=np.array(0)
    return r_tau
# print('r_tau',r_tau)
# Create a new model

# Create variables
def create_variable(Nr,Ne,M,D_max,F_i0,r_tau,Cs,delta_x):
    x=0
    for i in range(Nr):
        x=x+(F_i0[i]*(r_tau[i]*D_max + X_s))
    y=0
    for i in range(Nr):
        # delta_x=generate_delta_x()
        dum=0.5*(np.log(delta_x[i]))*((r_tau[i])**2)
        y=y+dum
    z=0
    for i in range (Ne+M):
        z=z+(r_tau[i]**2)
    # w=0
    # for i in range (0,Ne+M):
    #     w=w+(lam*(r_tau[i])-Cs)
    return x,y,z

def generate_solution(N,rho_ij_s,p_ij,h_ij,sigma_squared,k):
    r_tau_s=[0]*(N)
    s1=np.array(0)

    for i in range(N):
        r_tau_s[i]=B*(np.sum(np.array(rho_ij_s[i*k:(i+1)*k])*np.log2(1 + ((p_ij * h_ij) / sigma_squared))))
    r_tau_mean=np.mean(r_tau_s)
    return r_tau_mean,r_tau_s

# print("r_tau_s=",r_tau_s)
y_it=[0]*Nr
F_it=[0]*Nr
# for time_slot in range(0, num_time_slots ):
r_tau_f_embb=[]
r_tau_f_urllc=[]
transmission_delay_urllc_f=[]

for time in range(num_time_slots):
    if time > 2*num_time_slots/3:
        Ne=5
    rho_ij=generate_rho_ij(Ne+M+Nr,k)
    h_ij=generate_h_ij()
    # h_ij=0.01
    r_tau=generate_r_tau(Ne+M+Nr,k,p_ij,sigma_squared,B,h_ij,rho_ij)
    delta_x=generate_delta_x()
    x,y,z=create_variable(Nr,Ne,M,D_max,F_i0,r_tau,Cs,delta_x)
    m.setObjective(x+y-z)
    for j in range(k):
        m.addConstr(gp.quicksum(rho_ij[i, j] for i in range(Ne + Nr + M)) <= 1, f'prb_assignment_{j}')
    t=0
    for i in range(Ne+M+Nr):
        for j in range(k):
            t=rho_ij[i,j]+t
    m.addConstr(t <= k)
    m.addConstr(t >= 0.001)    
    for i in range(Ne+M+Nr):
        t=0
        for j in range(k):
            t=rho_ij[i,j]+t
        m.addConstr(t>=0.001)
    r_tau_sum=[0]
    for i in range (Ne+M):
        r_tau_sum[0] =r_tau[i]+r_tau_sum[0]
    m.addConstr((r_tau_sum[0])-(Cs)<= -0.001)
    m.params.NonConvex=2
    m.optimize()
    rho_ij_s=m.getAttr('x')
    # m.printAttr('x')
    # print(m.display())

    rho_ij_matrix = np.array([rho_ij_s[i * k:(i + 1) * k] for i in range(Ne + M + Nr)])
    print(f"time={time + 1} : rho_ij=\n{rho_ij_matrix}")
    # for v in m.getVars():
    #     print(f"{v.VarName}={v.X}")
    r_tau_mean_embb,r_tau_s_embb = generate_solution(Ne+M,rho_ij_s,p_ij,h_ij,sigma_squared,k)
    r_tau_mean_urllc,r_tau_s_urllc = generate_solution(Nr,rho_ij_s,p_ij,h_ij,sigma_squared,k)

    r_tau_f_embb.append(r_tau_mean_embb)
    r_tau_f_urllc.append(r_tau_mean_urllc)
    # r_tau_f_embb_meg=np.array(r_tau_f_embb) / 1e6
    # r_tau_f_urllc_meg=np.array(r_tau_f_urllc) / 1e6

    for i in range(Nr):
        y_it[i]=np.exp((r_tau_s_urllc[i]-ai)*D_max)-(1-X_s)
        F_i0[i]=np.maximum((F_i0[i]+y_it[i]),0)
    average_waiting_time_urllc = 1 / (r_tau_mean_urllc-ai)
    transmission_delay_urllc = (ai * average_waiting_time_urllc)
    transmission_delay_urllc_f.append(transmission_delay_urllc)
    # packet_length = np.random.exponential(scale=1/average_packet_size)  
    # transmission_delay_urllc = packet_length / r_tau_mean_urllc
    # total_system_delay += transmission_delay_urllc
plt.subplot(1,3,1)
plt.plot(r_tau_f_embb,label="eMBB")
plt.title('Mean Data_rate_eMBB')
plt.subplot(1,3,2)
plt.plot(r_tau_f_urllc, label="URLLC")
plt.title('Mean Data_rate_URLLC')
plt.subplot(1,3,3)
plt.plot(transmission_delay_urllc_f,label='Transmission Delay_URLLC')
# plt.ylim(0,2)

plt.legend()
plt.show()

    
    

