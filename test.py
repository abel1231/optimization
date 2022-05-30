from pyqpanda import *
import numpy as np
from portfolio_optimization import data_preprocessing
import time

def oneCircuit(qlist, Hamiltonian, beta, gamma):
    vqc=VariationalQuantumCircuit()
    for i in range(len(Hamiltonian)):
        tmp_vec=[]
        item=Hamiltonian[i]
        dict_p = item[0]
        for iter in dict_p:
            if 'Z'!= dict_p[iter]:
                pass
            tmp_vec.append(qlist[iter])

        coef = item[1]

        if 2 == len(tmp_vec):
            vqc.insert(VariationalQuantumGate_CNOT(tmp_vec[0], tmp_vec[1]))
            vqc.insert(VariationalQuantumGate_RZ(tmp_vec[1], 2*gamma*coef))
            vqc.insert(VariationalQuantumGate_CNOT(tmp_vec[0], tmp_vec[1]))
        elif 1 == len(tmp_vec):
            vqc.insert(VariationalQuantumGate_RZ(tmp_vec[0], 2 * coef * gamma))
        else:
            raise AssertionError()


    for j in qlist:
        vqc.insert(VariationalQuantumGate_RX(j,2.0*beta))
    return vqc

file_path = "./data/stock_data.xlsx"
exp_ret, cov_mat = data_preprocessing(file_path)
exp_ret = exp_ret.to_numpy()
cov_mat = cov_mat.to_numpy()


budget = 1
# Gf = 1.0 / (2 ** (g - 1)) # granularity
Gf = 1.0
theta1 = Gf
theta2 = 2.5 * Gf * Gf
theta3 = 1.0
num_assets = 6
epoch = 100

num_qubit = num_assets
machine = init_quantum_machine(QMachineType.CPU)

qlist = machine.qAlloc_many(num_qubit)

J = (theta3 * budget**2 * Gf**2 + theta2 * cov_mat) / 4
h = (-theta1 * exp_ret - 2*theta3* budget**2 * Gf) / 2 + np.sum(J, axis=1)

problem = {}

for i in range(num_qubit):
    Pauli = 'Z' + str(i)
    problem[Pauli] = h[i]

for i in range(num_qubit):
    for j in range(i+1, num_qubit):
        Pauli = 'Z{:d} Z{:d}'.format(i, j)
        problem[Pauli] = J[i][j]

Hp = PauliOperator(problem)
print(Hp)
step = 2

beta = var(np.ones((step,1),dtype = 'float64'), True)
gamma = var(np.ones((step,1),dtype = 'float64'), True)

vqc=VariationalQuantumCircuit()

for i in qlist:
    vqc.insert(VariationalQuantumGate_H(i))

for i in range(step):
    vqc.insert(oneCircuit(qlist,Hp.toHamiltonian(1),beta[i], gamma[i]))


loss = qop(vqc, Hp, machine, qlist)
optimizer = MomentumOptimizer.minimize(loss, 0.01, 0.9)

leaves = optimizer.get_variables()

loss_value_his = 0
loss_value_min = 0
count = 0
start = time.time()
for i in range(epoch):
    if count > 20:
        break
    start_local = time.time()
    optimizer.run(leaves, 0)
    loss_value = optimizer.get_loss()
    if loss_value_min > loss_value:
        loss_value_min = loss_value
    if np.abs(loss_value - loss_value_his) < 1e-4 or loss_value > loss_value_min:
        count += 1
    else:
        count = 0
    loss_value_his = loss_value
    print("epoch:", i, " loss: {:.8f}".format(loss_value), " time: {:.2f}s".format(time.time() - start_local))

print("Training done! Total elapsed time:{:.2f}s".format(time.time()-start))
# 验证结果
prog = QProg()
qcir = vqc.feed()
prog.insert(qcir)
directly_run(prog)

result = quick_measure(qlist, int(1e6))
print(sorted(result.items(), key = lambda kv:(kv[1], kv[0]), reverse=True))