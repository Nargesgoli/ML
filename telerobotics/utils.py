import numpy as np
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from math import log
def optim(num_time_slots,Ne,M,Nr,k,p_ij,sigma_squared,B,h_ij,delta_x2,D_max,F_i0,Cs_e,ai,delx,time,X_s,Cs_r):
    
        m = gp.Model()
        rho_ij=generate_rho_ij(m,Ne+M+Nr,k)
        rho_ij_s=[0]*((Ne+Nr+M)*k)
        # h_ij=generate_h_ij()
        # h_ij=0.01
        r_tau=generate_r_tau(Ne,M,Nr,k,p_ij,sigma_squared,B,h_ij,rho_ij)
        # delta_x=generate_delta_x(1)
        delta_x=delta_x2
        delx.append((delta_x))
        # delx.append(np.mean(delta_x))
        X,Y,Z=create_variable(Ne,M,Nr,D_max,F_i0,r_tau,Cs_e,delta_x,ai,X_s)
        # print('Y=',Y)
        m.setObjective(X+Y-Z)
        # m.setObjective(Y-Z)
        for j in range(k):
            col_sum = 0
            for i in range(Ne +M+ Nr ):
                col_sum += rho_ij[i,j]
            m.addConstr(col_sum <= 1,f'prb_assignment_{j}')
            # m.addConstr(gp.quicksum(rho_ij[i, j] for i in range(Ne + Nr + M)) <= 1, f'prb_assignment_{j}')
        t=0
        for i in range(Ne+M+Nr):
            for j in range(k):
                t=rho_ij[i,j]+t
        m.addConstr(t ==k, f'using all PRBs')
        # m.addConstr(t >= 0.001)    
        
        for i in range(Ne+M+Nr):
            t1=0
            for j in range(k):
                t1=rho_ij[i,j]+t1
            m.addConstr(t1 >=  0.001,f'each user uses at least one PRB')
        # row_sum=[]
        # for i in range(Ne+M):
        #     row_sum1=0
        #     for j in range(k):
        #         row_sum1 += rho_ij[i,j]
        #     row_sum.append(row_sum1)
        # for d in range(Ne+M):
        #     for g in range (d+1,Ne+M):
        #         m.addConstr( (row_sum[d]-row_sum[g])<= 3,f'equal distribution1')
        #         m.addConstr( (row_sum[d]-row_sum[g]) >= -3,f'equal distribution2')

        r_tau_sum_e=[0]
        for v in range (Ne+M):
            r_tau_sum_e[0] =r_tau[v]+r_tau_sum_e[0]
        m.addConstr((r_tau_sum_e[0])-(Cs_e)<= -0.001)
        # # r_tau_sum_r=[0]
        # for v1 in range (Nr):
        #     r_tau_sum_r[0] =r_tau[Ne+M+v1]+r_tau_sum_r[0]
        # m.addConstr((r_tau_sum_r[0])-(Cs_r)<= -0.001)   
        m.params.NonConvex=2
        m.optimize()
        rho_ij_s=m.getAttr('x')
        m.printAttr('x')
        # print(m.display())
        obj = m.getObjective()
        print(obj.getValue())

        rho_ij_matrix = np.array([rho_ij_s[i * k:(i + 1) * k] for i in range(Ne + M + Nr)])
        print(f"time={time + 1} : rho_ij=\n{np.reshape(rho_ij_s,(-1,k))}")
        # for v in m.getVars():
        #     print(f"{v.VarName}={v.X}")
        return rho_ij_s
        
        
def postprocess(Ne,Nr,M,rho_ij_s,p_ij,h_ij,sigma_squared,k,B,r_tau_f_embb,r_tau_f_urllc,ai,D_max,X_s,y_it,F_i0,transmission_delay_urllc_f,average_packet_size,total_system_delay):        
        r_tau_mean_embb,r_tau_s_embb,r_tau_mean_urllc,r_tau_s_urllc,r_tau_s = generate_solution(Ne,M,Nr,rho_ij_s,p_ij,h_ij,sigma_squared,k,B)
        # r_tau_mean_urllc,r_tau_s_urllc = generate_solution(Ne,M,Nr,rho_ij_s,p_ij,h_ij,sigma_squared,k)

        r_tau_f_embb.append(r_tau_mean_embb)
        r_tau_f_urllc.append(r_tau_mean_urllc)
        # r_tau_f_embb_meg=np.array(r_tau_f_embb) / 1e6
        # r_tau_f_urllc_meg=np.array(r_tau_f_urllc) / 1e6

        for l in range(Nr):
            y_it=np.exp((r_tau_s_urllc[l]-ai)*D_max)-(1-X_s)
            F_i0[l]=np.maximum((F_i0[l]+y_it),0)
        average_waiting_time_urllc = 1 / ((r_tau_mean_urllc-ai)*r_tau_mean_urllc)
        transmission_delay_urllc = (ai * average_waiting_time_urllc)
        transmission_delay_urllc_f.append(transmission_delay_urllc)
        packet_length = np.random.exponential(scale=1/average_packet_size)  
        transmission_delay_urllc = packet_length / r_tau_mean_urllc
        total_system_delay += transmission_delay_urllc
        return transmission_delay_urllc_f,transmission_delay_urllc_f,r_tau_f_embb,r_tau_f_urllc,total_system_delay
        


def generate_solution(Ne,M,Nr,rho_ij_s,p_ij,h_ij,sigma_squared,k,B):
    sum1= Ne +M + Nr
    r_tau_s=[0]*(sum1)
    # r_tau_s_embb=[0]*(Ne+M)
    # r_tau_s_urllc=[0]*(Nr)
    for i in range(Ne+M):
        # r_tau_s_embb[i]=B*(np.sum(np.array(rho_ij_s[i*k:(i+1)*k])*np.log2(1 + ((p_ij * h_ij) / sigma_squared))))
        r_tau_s[i]=B*(np.sum(np.array(rho_ij_s[i*k:(i+1)*k])*np.log2(1 + ((p_ij * h_ij) / sigma_squared))))
    # r_tau_mean_embb=np.mean(r_tau_s_embb)
    
    for i in range(Ne+M,sum1):
        # r_tau_s_urllc[i]=B*(np.sum(np.array(rho_ij_s[i*k:(i+1)*k])*np.log2(1 + ((p_ij * h_ij) / sigma_squared))))
        r_tau_s[i]=B*(np.sum(np.array(rho_ij_s[i*k:(i+1)*k])*np.log2(1 + ((p_ij * h_ij) / sigma_squared))))
        # r_tau_s_urllc[i]=r_tau_s[i+Ne+M]
    r_tau_s_embb=r_tau_s[:Ne+M]
    r_tau_s_urllc=r_tau_s[Ne+M:Ne+M+Nr]
    r_tau_mean_embb=np.mean(r_tau_s_embb)
    r_tau_mean_urllc=np.mean(r_tau_s_urllc)    
    return r_tau_mean_embb,r_tau_s_embb,r_tau_mean_urllc,r_tau_s_urllc,r_tau_s

def create_variable(Ne,M,Nr,D_max,F_i0,r_tau,Cs_e,delta_x,ai,X_s):
    xt=0
    for i in range(Nr):
        dum1= 10 *(F_i0[i]*((((r_tau[i+Ne+M])-ai)*D_max) + X_s))
        xt=xt + dum1
        # print('xt=',xt)
    y=0
    for i in range(Nr):
        # delta_x=generate_delta_x()
        # dum2=0.1 * (np.abs((np.abs(delta_x))))*(((r_tau [i+Ne+M])**2))
        dum2=1 * (np.abs((np.abs(delta_x))))*(((r_tau [i+Ne+M])**2))

        y= y+ dum2
        # print('y=',y)
    z=0
    for i in range (Ne+M):
        dum3=0.001*((r_tau[i])**2)
        z= z +dum3
        # print('z=',z)
    return xt,y,z

def generate_r_tau( Ne, M , Nr,k,p_ij,sigma_squared,B,h_ij,rho_ij):
    sum=Ne+M+Nr
    r_tau=list([0]*(Ne+M+Nr))
    r_tau_embb=list([0]*(Ne+M))
    r_tau_urllc=list([0]*(Nr))
    s_embb=np.array(0)
    s_urllc=np.array(0)
    for i in range(Ne+M):
        for j in range(k ):
            s_embb=s_embb+(B * (rho_ij[i, j] * (np.log2(1 + (p_ij *h_ij) / (sigma_squared)))))
        # r_tau_embb[i]=s_embb
        r_tau[i]=s_embb
        s_embb=np.array(0)
    for i in range(Ne+M,sum):
        for j in range(k ):
            s_urllc=s_urllc+(B * (rho_ij[i, j] * (np.log2(1 + (p_ij *h_ij) / (sigma_squared)))))
        # r_tau_urllc[i]=s_urllc
        r_tau[i]=s_urllc
        s_urllc=np.array(0)
    return r_tau
def generate_rho_ij(m,sum,k):
    rho_ij={}
    for i in range(sum):
        for j in range(k):
            rho_ij[i, j] = m.addVar(name=f'rho_{i}_{j}', vtype=GRB.BINARY)
    return rho_ij

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
def generate_delta_x(Nu):
    mu, sigma = 10, 5  # Mean and standard deviation
    delta_x_normal = np.random.normal(mu, sigma, 1000)
    delta_x_normal_binary = np.clip(delta_x_normal, 0, 20)
    delta_x_random=np.random.choice(delta_x_normal_binary,size=Nu)
    # print("delta_x=",delta_x) 
    return delta_x_random+0.00000001 
def generate_delta_x2(num_time_slots):
    intervals = 5
    values_per_interval = int(num_time_slots/intervals)
    sharp_change_factor = 5  # Factor to control the sharpness of the change

    delta_x_values = []

    for _ in range(intervals):
        mu_delta_x, sigma_delta_x = np.random.uniform(5, 15), np.random.uniform(5, 15)
        mu_sharp, sigma_sharp = np.random.uniform(5, 15), np.random.uniform(5, 15)

        # Generate a random value for delta_x within each interval
        delta_x_normal = np.random.normal(mu_delta_x, sigma_delta_x)
        current_value = np.clip(delta_x_normal, 0, 10) + 0.001

        # Append the current value multiple times for the interval
        delta_x_values.extend([current_value] * values_per_interval)

        # Generate a random value for sharp_change within each interval
        sharp_change = np.random.normal(mu_sharp, sigma_sharp)
        sharp_change = np.clip(sharp_change * sharp_change_factor, 0, 20)
        delta_x_values.extend([sharp_change] * values_per_interval)

    return np.array(delta_x_values)+1
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
