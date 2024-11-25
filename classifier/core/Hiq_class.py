import matplotlib.pyplot as plt
import numpy as np
from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.gates import H, RX,X,Rzz
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.framework import MQAnsatzOnlyLayer
from mindquantum.simulator import Simulator
import networkx as nx
import mindspore.nn as nn
import mindspore as ms
import time
from .inner_to_Eu import *

def draw_g(data_list):
    distances_to_centroids=Eu_circuit(data_list,shots_n=10000,seeds=6).calculate_distance(7,'infidelity')
    # distances_to_centroids=Eu(data_list)
    g = nx.Graph()
    g.add_weighted_edges_from(distances_to_centroids)
    labels = nx.get_edge_attributes(g, 'weight')
    sum_weights=sum(labels.values())
    return g,labels,sum_weights 
def build_hc(g, para):
    hc = Circuit()                  
    for i,j in enumerate(g.edges):
        hc += Rzz(f'{para}{i}').on(j)        
    hc.barrier()                    
    return hc

def build_hb(g, para):
    hb = Circuit()                  
    for j in g.nodes:
        hb += RX(para+str(j)).on(j)       
    hb.barrier()                    
    return hb

def build_ansatz(g, p):                    
    circ = Circuit()                       
    for i in range(p):
        circ += build_hc(g, f'g{i}_')       
        circ += build_hb(g, f'b{i}_')       
    return circ
def build_ham(g,labels):
    ham = QubitOperator()
    edge_weights=labels
    for edge,weight in edge_weights.items():   
        ham += QubitOperator(f'Z{edge[0]} Z{edge[1]}',weight) 
    ham=Hamiltonian(ham)
    return ham
def classifer(data_list,y):
    g,labels,sum_weights =draw_g(data_list)
    d = 3  
    ham=build_ham(g,labels=labels)
    init_state_circ = UN(H, g.nodes)            
    ansatz = build_ansatz(g, d)                 
    circ = init_state_circ + ansatz              
    
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    sim = Simulator('mqvector', circ.n_qubits)                     
    grad_ops = sim.get_expectation_with_grad(ham, circ)            
    net = MQAnsatzOnlyLayer(grad_ops)                              
    opti = nn.Adam(net.trainable_params(), learning_rate=0.1)     
    train_net = nn.TrainOneStepCell(net, opti)                     
    losses = []
    steps=[]
    for i in range(200):    
        t_n=train_net()
        cut =(sum_weights - t_n) / 2 
        losses.append(t_n)
        steps.append(i)
    pr=dict(zip(ansatz.params_name, net.weight.asnumpy())) 
    circ.measure_all()                               
    shots_n=1000
    sample=sim.sampling(circ, pr=pr, shots=shots_n)
    sample_bit_string=sample.bit_string_data

    y=np.array(y[::-1])
    len_y=len(y)-2   
    y_0 = ''.join(y.flatten().astype(str))   
    result=y_0
    y_1 = ''.join(['1' if c == '0' else '0' for c in y_0])  
    
    p=0
    for i,j in sample_bit_string.items():
        if i[:2]==y_0[:2]: 
            matching_count = sum(a == b for a, b in zip(i[2:], y_0[2:]))  
            matching_count_p=matching_count/len_y  
            p+=j/shots_n*matching_count_p
        elif i[:2]==y_1[:2]:
            matching_count = sum(a == b for a, b in zip(i[2:], y_1[2:]))
            matching_count_p=matching_count/len_y
            p+=j/shots_n*matching_count_p
    print(p)
    return p,losses,steps
if __name__=="__main__":
    time_start=time.time()
    data_list=np.array([[0.354,0.935],[0.987,0.159],[0.999,0.035],[0.147,0.989],[0.338,0.941],[0.997,-0.072],[0.999,-0.032],[-0.439,0.899],[0.987,-0.161],[0.173,0.985]])
    draw_g(data_list)
    y=[1,0,0,1,1,0,0,1,0,1]
    acc=classifer(data_list=data_list,y=y)
    print("Accuracy=",acc)
    time_span = time.time() - time_start
    print('Run time:', time_span)
    



