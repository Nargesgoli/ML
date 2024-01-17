### investigating the impact oh Delat_x

# import gurobipy

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from math import log
from utils import optim
from utils import generate_h_ij
from utils import generate_delta_x
from utils import generate_delta_x2
from utils import postprocess
# Define constants
B = 180e-3
sigma_dBm = -173.9
sigma_squared = (10**((sigma_dBm - 30) / 10.0))  # Convert dBm to watts
# sigma_squared=-173.9
k = 12 # #number of PRBs
p_ij_dBm= 10 #*********************************
p_ij= (10**((p_ij_dBm-30) / 10.0))

D=0.2
Ne=2
Nr=2
M=1
F_i0=[0]*Nr
X_s=0.95
time_slot_duration = 1  # 100 milliseconds
D_max=0.01
# D_max=time_slot_duration
Cs=100

Cs_e=200
Cs_r=50
# Cs_e=400
# Cs_r=100
ai=1
average_packet_size = 10e-3
# Simulation parameters
num_time_slots = 50
time_slot_duration = 1  # 100 milliseconds
delx=[]
r_tau_f_embb=[]
r_tau_f_urllc=[]
y_it=[]
transmission_delay_urllc_f=[]
total_system_delay=0
h_ij=generate_h_ij()
delta_x2=generate_delta_x2(num_time_slots)
DeltaX=[]
for time in range(num_time_slots):   
        # if time>2:
    # h_ij=generate_h_ij()
    h_ij= 0.01
    # delta_x2=generate_delta_x(1)
    DeltaX.append(delta_x2[time])
    if time > (7*num_time_slots/10):
        Ne=5
        print("pause")
    # sol=optim(num_time_slots,Ne,M,Nr,k,p_ij,sigma_squared,B,h_ij,delta_x2[time],D_max,F_i0,Cs_e,ai,delx,time,X_s,Cs_r)
    sol=optim(num_time_slots,Ne,M,Nr,k,p_ij,sigma_squared,B,h_ij,delta_x2[time],D_max,F_i0,Cs_e,ai,delx,time,X_s,Cs_r)

    transmission_delay_urllc_f,transmission_delay_urllc_f,r_tau_f_embb,r_tau_f_urllc,total_system_delay=postprocess(Ne,Nr,M,sol,p_ij,h_ij,sigma_squared,k,B,r_tau_f_embb,r_tau_f_urllc,ai,D_max,X_s,y_it,F_i0,transmission_delay_urllc_f,average_packet_size,total_system_delay)
    

# Generate Rayleigh gain samples
# h_ij=generate_h_ij()




# lam= m.addVar(name=f'lambda',vtype=GRB.CONTINUOUS)



# Create variables




# print("r_tau_s=",r_tau_s)
# rho_ij_s=[0]
# y_it=[0]*Nr
# F_it=[0]*Nr
# # for time_slot in range(0, num_time_slots ):
# r_tau_f_embb=[]
# r_tau_f_urllc=[]
# transmission_delay_urllc_f=[]
# delx=[]
# delta_x2=generate_delta_x2(num_time_slots)
# h_ij=generate_h_ij()
plt.subplot(2,2,1)
plt.plot(r_tau_f_embb,label="eMBB")
# plt.plot([Cs]*num_time_slots,label='Capacity',linestyle='dashed')
plt.title('Mean Data_rate_eMBB')
plt.xlabel('time[s]')
plt.ylabel('$R_i-eMBB(Mbps)$')
# plt.ylim(0,np.max(r_tau_f_embb)+5)
plt.subplot(2,2,2)
plt.plot(r_tau_f_urllc, label="URLLC")
plt.title('Mean Data_rate_URLLC')
plt.xlabel('time[s]')
plt.ylabel('$R_i-URLLC(Mbps)$')
# plt.ylim(0,np.max(r_tau_f_urllc)+5)
plt.subplot(2,2,3)
# plt.plot([D_max]*num_time_slots,label='D_max',linestyle='dashed')
plt.plot(transmission_delay_urllc_f,label='Transmission Delay_URLLC')
plt.plot([D_max]*num_time_slots, 'r--',label='Delay Threshold' )
plt.legend()
# plt.title('transmission_delay_urllc')
plt.xlabel('time(s)')
plt.ylabel('Delay_URLLC(s)')
plt.subplot(2,2,4)
plt.plot(DeltaX,label='$\Delta x$')
# plt.title('$\Delta_x Deviation$')
# plt.ylim(0,2)
plt.xlabel('time(s)')
plt.ylabel('$\Delta x$(in)')
plt.legend()
plt.show()

    
    

