import cmath
import numpy as np
from mindquantum.simulator import Simulator
from mindquantum.core.circuit import Circuit, UN,controlled,dagger
from mindquantum.core.gates import H, RX,X,Rzz,RY,PhaseShift,SWAP,Measure
class Eu_circuit:
    def __init__(self,data,shots_n,seeds):
        self.data=data
        self.parameters=self.make_para()
        self.sample_n=len(self.parameters)
        self.inner_product_list,self.fidelity_list,self.infidelity_list=self.inner_product(shots_n,seeds)
        self.edge_n=len(self.inner_product_list)
    
        self.funcs_inn={                  # F represents the inner product and its value is [-1,1]
            1:lambda F:2/(F+1),           #[1,+inf) reciprocal
            2:lambda F:np.log2(2/(F+1)),     #[0,+inf)
            3:lambda F:np.log2(3/(F+2)),      #[0,ln3]
            4:lambda F:np.exp(2/F+1),          #[e,+inf)
            5:lambda F:1/2*(F+1),              #fidelity   [0,1]
            6:lambda F:1/2*(1-F),              #infidelity [1,0]
            7:lambda F:arccos(F),              #[pi,0]
            8:lambda F:np.sqrt(2*(1-F))        #Euclidean distance
        }
        self.funcs_fid={                  #F represents fidelity, and its value range is [0,1]
            1:lambda F:1/F,               #[1,+inf) reciprocal
            2:lambda F:np.log(1/F),      #[0,+inf)
            3:lambda F:np.log2(2/(F+1)),      #[0,ln2]
            4:lambda F:np.log2((1+np.sqrt(1-F**2))/F),  #[0,+inf)
            5:lambda F:F,                #fidelity   [0,1]
            6:lambda F:1-F,              #infidelity [1,0]
            7:lambda F:2*np.sqrt(1-F),        #Euclidean distance
            8:lambda F:arccos(F),          #[pi/2,0]
        }
        self.funcs_infid={               #F represents infidelity, and its value range is [0,1]
            1:lambda F:F**2,            #[0,1] reciprocal
            2:lambda F:np.log2(F+1),     #[0,ln2]
            3:lambda F:exp(F),          #[1,e]
            4:lambda F:arcsin(F),       #[0,pi/2]
            5:lambda F:F,               #infidelity   [0,1]
            6:lambda F:1-F,              #fidelity   [1,0]
            7:lambda F:2*np.sqrt(F),           #Euclidean distance
            8:lambda F:np.sin(F)               #[0,sin1]
        }
    def make_para(self):
        r=[]
        parameters=[]
        for i in self.data:
            para=self.prepare_para(i)
            parameters.append(para)
            r.append([np.exp(para[2]*1j)*np.cos(para[0]/2),np.exp(para[1]*1j)*np.sin(para[0]/2)])
        result=np.round(r,3)
        return parameters
    def prepare_para(self,vector):
        alpha=cmath.phase(vector[0])
        varphi=cmath.phase(vector[1])
        theta=2*np.arctan(abs(vector[1]/vector[0]))
        return [theta,varphi,alpha]
    #RZ PS PS encoder
    def angle_encoder(self,para_abc,qubit):
        circ = Circuit(RY(para_abc[0]).on(qubit)) 
        circ += PhaseShift(para_abc[1]).on(qubit)  
        circ += X.on(qubit)  
        circ += PhaseShift(para_abc[2]).on(qubit)  
        circ += X.on(qubit)  
        return circ
    def measure_circuit(self,centroid,sample):
        circ = Circuit()
        circ += self.angle_encoder(centroid,3)
        circ.barrier()
        circ+=H.on(1)
        circ+=H.on(2)
        circ.barrier()
        circ+=dagger(controlled(self.angle_encoder(centroid,3))(2))
        circ.barrier()
        circ+=controlled(self.angle_encoder(sample,3))(2)
        circ.barrier()
        circ+=H.on(0)
        circ += SWAP.on([1,2],0)
        circ+=H.on(0)
        circ+=Measure('q0').on(0) 
        return circ
    def inner_product(self,shots_n,seeds):
        inn_list=[]
        fid_list=[]
        infid_list=[]
        sim=Simulator('mqvector',4)
        for i in range(2):
            for j in range(2,self.sample_n):
                sim.reset()
                circ=self.measure_circuit(self.parameters[i],self.parameters[j])
                result=sim.sampling(circ,shots=shots_n,seed=seeds).bit_string_data['0']/shots_n
                P=4*result-3
                inn_list.append([i,j,P])
                fid_list.append([i,j,1/2*(P+1)])
                infid_list.append([i,j,1/2*(1-P)])
        sim.reset()
        circ=self.measure_circuit(self.parameters[0],self.parameters[1])
        result=sim.sampling(circ,shots=shots_n,seed=seeds).bit_string_data['0']/shots_n
        P=4*result-3
        inn_list.append([0,1,P])
        fid_list.append([0,1,1/2*(P+1)])
        infid_list.append([0,1,1/2*(1-P)])
        return inn_list,fid_list,infid_list
    def calculate_distance(self,select_fuc,option='infidelity',options=['inner','fidelity','infidelity']):
        if option not in options:
            raise ValueError(f"Invalid option. Choose from {options}.")
        distance_list=[]
        if option=='inner':
            for i,j,k in self.inner_product_list:
                distance_list.append([i,j,self.funcs_inn[select_fuc](k)])
        elif option=='fidelity':
            for i,j,k in self.fidelity_list:
                distance_list.append([i,j,self.funcs_fid[select_fuc](k)])
        else:
            for i,j,k in self.infidelity_list:
                distance_list.append([i,j,self.funcs_infid[select_fuc](k)])
        distance_list[-1][-1]*=(self.sample_n-2)
        return distance_list
from scipy.spatial.distance import euclidean
def Eu(data):
    n=len(data)
    result=[]
    for i in range(2):
        for j in range(2,n):
            result.append([i,j,euclidean(data[i],data[j])])
    result.append([0,1,(n-2)*euclidean(data[0],data[1])])
    return result

if __name__=="__main__":
    data_list=[[0.354,0.935],[0.987,0.159],[0.999,0.035],[0.147,0.989],[0.338,0.941],[0.997,-0.072],[0.999,-0.032],[-0.439,0.899],[0.987,-0.161],[0.173,0.985]]
    print("Euclidean:",Eu(data_list[:]))
    model=Eu_circuit(data_list,shots_n=10000,seeds=6)
    inn_to_Eu=model.calculate_distance(7)
    # print("inner product:",model.inner_product_list)
    # print("fidelity:",model.fidelity_list)
    # print("infidelity:",model.infidelity_list)
    print("Euclidean:",inn_to_Eu)


