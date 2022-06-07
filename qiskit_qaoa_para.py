import time
import argparse

from qiskit import QuantumCircuit, transpile
from qiskit import Aer
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit import Parameter
from qiskit.providers.aer import StatevectorSimulator
#from portfolio_optimization import data_preprocessing
import numpy as np
from qiskit.opflow import PauliSumOp
from scipy.optimize import minimize
import pandas as pd

def data_preprocessing(file_path):
    '''
    Return expected return and covariance matrix of the dataset
    '''
    ## excel --> array
    df = pd.read_excel(file_path)
    data = df.values
    ## closing price --> rate of return
    RoR = np.zeros([data.shape[0]-1,data.shape[1]])   # Rate of return
    for i in range(data.shape[0]-1):
        RoR[i,:] = (data[i+1,:]-data[i,:])/data[i,:]
    ## Calculate the expected return and covariance matrix
    # RoR2 = np.hstack((RoR, RoR))
    # data_df = pd.DataFrame(RoR2)
    data_df = pd.DataFrame(RoR)
    exp_ret = data_df.mean() #* data.shape[0]
    cov_mat = data_df.cov() #* data.shape[0]
    return exp_ret, cov_mat

def calc_J():
    '''
    calculate the coefficients of all rzz gates
    :return: a coefficient matrix
    '''
    J = np.zeros((num_qubits, num_qubits))
    for i in range(num_assets):
        for j in range(num_assets):
            for k1 in range(num_slices):
                for k2 in range(num_slices):
                    J[i*num_slices + k1][j*num_slices + k2] = 2**(k1+k2-2) * (theta2 * cov_mat[i][j] + theta3 * budget**2 * Gf**2)
    return J * 2

def calc_h():
    '''
    calculate the coefficients of all rz gates
    :return: a coefficient vector
    '''
    h = np.zeros(num_qubits)
    seq = [2 ** (k - 1) for k in range(num_slices)]
    con1 = np.sum(np.array(seq))
    seq = [2 ** (k + 1) for k in range(num_slices)]
    con2 = np.sum(np.array(seq))
    for i in range(num_assets):
        for k in range(num_slices):
            h[i * num_slices + k] = 2 ** (k - 1) * (theta1 * exp_ret[i] -
                                                    2 * theta3 * Gf * budget ** 2 * (num_assets * Gf * con1 - 1) -
                                                    theta2 / 4.0 * con2 * (np.sum(cov_mat, axis=1)[i] + np.sum(cov_mat, axis=0)[i]))
    return h

def insert_RX(beta):
    qc = QuantumCircuit(num_qubits)
    for i in range(0, num_qubits):
        qc.rx(2 * beta, i)
    return qc

def insert_RZ(gamma, h):
    qc = QuantumCircuit(num_qubits)
    for i in range(0, num_qubits):
        qc.rz(2 * h[i] * gamma, i)
    return qc

def insert_RZZ(gamma, J):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            qc.rzz(2 * J[i][j] * gamma, i, j)
    qc.barrier()
    return qc

def insert_H():
    qc = QuantumCircuit(num_qubits)
    for i in range(0, num_qubits):
        qc.h(i)
    return qc

def get_Pauli(index, type):
    if type == 'Z':
        assert len(index) == 1
        index = index[0]
        assert index >= 0 and index <= num_qubits - 1
        _Pauli = ['I'] * (num_qubits - 1)
        _Pauli.insert(index, 'Z')
        _Pauli = ''.join(_Pauli)
        return _Pauli
    elif type == 'ZZ':
        assert len(index) == 2
        _Pauli = ['I'] * (num_qubits - 2)
        for i in range(len(index)):
            assert index[i] >= 0 and index[i] <= num_qubits - 1
            _Pauli.insert(index[i], 'Z')
        _Pauli = ''.join(_Pauli)
        return _Pauli
    else:
        raise AssertionError()

def problem_PauliOperator(h, J):
    Pauli_h_list = []
    for i in range(num_qubits):
        Pauli_h_list.append((get_Pauli([i], 'Z'), h[i]))
    Pauli_h = PauliSumOp.from_list(Pauli_h_list, coeff=1.0)

    Pauli_J_list = []
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            Pauli_J_list.append((get_Pauli([i,j], 'ZZ'), J[i][j]))
    Pauli_J = PauliSumOp.from_list(Pauli_J_list, coeff=1.0)

    Pauli_sum = Pauli_h + Pauli_J

    return Pauli_h, Pauli_J, Pauli_sum

def oneCircuit(h, J, beta, gamma):
    qc = QuantumCircuit(num_qubits)
    qc.append(insert_RX(beta), [i for i in range(0, num_qubits)])
    qc.append(insert_RZ(gamma, h), [i for i in range(0, num_qubits)])
    qc.append(insert_RZZ(gamma, J), [i for i in range(0, num_qubits)])
    return qc

def get_expectation(circuit, para_list, Hamiltonian):

    def execute_circ(theta):
        qc = QuantumCircuit(num_qubits)

        p = len(theta) // 2
        beta = theta[:p]
        gamma = theta[p:]

        para_dict = {}
        for i in range(p):
            para_dict[para_list[i]] = beta[i]
            para_dict[para_list[i+p]] = gamma[i]
        qc.append(circuit, [i for i in range(0, num_qubits)])
        qc.assign_parameters(para_dict, inplace=True)
        circ = transpile(qc, simulator)
        result = simulator.run(circ).result()
        _statevector = result.get_statevector(circ)  # innner product of statevector_dagger and statevector is 1
        statevector = np.array(_statevector)
        statevector_dagger = np.array(_statevector.conjugate())
        loss =  statevector_dagger @ Hamiltonian @ statevector
        assert np.imag(loss) < 1e-10
        return np.real(loss)

    return execute_circ

def str_to_statevector(string):
    string = string[::-1]
    dec = int(string, 2)
    state = np.zeros(2 ** len(string))
    state[dec] = 1.0
    return state[None,:]

def compute_utility(x_str):
    x = []
    for i in range(len(x_str)):
        x.append(int(x_str[i]))
    n = 6
    w = np.zeros(n, dtype=np.int32)
    for i in range(n):
        for j in range(num_slices):
            w[i] += x[i*num_slices+j]*(j+1)
    w_sum = sum(w)
    w = w*Gf
    utility = w@exp_ret-2.5*w@np.dot(w,cov_mat)
    return w_sum, utility

def print_result(circuit, Hamiltonian, para_list, solution):
    qc = QuantumCircuit(num_qubits)

    p = len(solution) // 2
    beta = solution[:p]
    gamma = solution[p:]

    para_dict = {}
    for i in range(p):
        para_dict[para_list[i]] = beta[i]
        para_dict[para_list[i + p]] = gamma[i]
    qc.append(circuit, [i for i in range(0, num_qubits)])
    qc.assign_parameters(para_dict, inplace=True)
    circ = transpile(qc, simulator)
    result = simulator.run(circ).result()
    statevector = result.get_statevector(circ)  # innner product of statevector_dagger and statevector is 1
    statevector = statevector.to_dict()
    a = 0
    for i in statevector:
        statevector[i] = np.abs(np.array(statevector[i])) ** 2
        a = a + statevector[i]
    # print('a: %f' % a)
    # print(statevector)
    result = sorted(statevector.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    # print(result)
    mm = []
    for i in range(len(result)):
        x, _ = result[i]
        mm.append(str_to_statevector(x))
    mm = np.concatenate(mm, axis=0)
    value_mm = np.sum((mm @ Hamiltonian) * mm, axis=1)

    min_index = np.argmin(value_mm)
    _, opt_utility = compute_utility(result[min_index][0][::-1])
    print("\nOptimal: selection {}, value {:.8f}, utility {:.8f}".format(result[min_index][0][::-1], value_mm[min_index], opt_utility))
    

    print("\n----------------------- Full result  ---------------------------")
    print("rank\tselection\tvalue\t\tutility\t\tprobability")
    print("------------------------------------------------------------------")
    for i in range(len(result)):
        x, probability = result[i]
        value = value_mm[i]
        assert np.imag(value) < 1e-10
        value = np.real(value)
        # value = portfolio.to_quadratic_program().objective.evaluate(x)
        w_sum, utility = compute_utility(x[::-1])        
        flag = True if utility >= opt_utility else False
        
        print("%d\t%-10s\t%.8f\t%.8f\t%s\t%d\t%.8f" % (i, x[::-1], value, utility, flag, w_sum, probability))


parser = argparse.ArgumentParser()
parser.add_argument('--budget', type=int, default= 8)
parser.add_argument('--theta3', type=int, default= 1/2)
parser.add_argument('--layers', type=int, default= 7)
parser.add_argument('--maxiter', type=int, default= 3000)
parser.add_argument('--num_assets', type=int, default= 6)
parser.add_argument('--num_slices', type=int, default= 2)
parser.add_argument('--rand_start', type=float, default= -0.1)
parser.add_argument('--rand_end', type=float, default= 0.1)
arg = parser.parse_args()    
    

# 初始化参数
budget = arg.budget
theta3 = arg.theta3
layers = arg.layers
maxiter = arg.maxiter
num_assets = arg.num_assets
num_slices = arg.num_slices
rand_start = arg.rand_start
rand_end = arg.rand_end

Gf = 1.0 / budget
theta1 = Gf
theta2 = 2.5 * Gf * Gf
# The number of binary bits required to represent one asset (g in the paper)
num_qubits = num_assets * num_slices

np.random.seed(12345)
# 读取收益和方差
file_path = "./data/stock_data.xlsx"
exp_ret, cov_mat = data_preprocessing(file_path)
exp_ret = exp_ret.to_numpy()
cov_mat = cov_mat.to_numpy()

# 计算所给问题对应的哈密尔顿量的系数
J = calc_J()
h = calc_h()

# 计算所给问题对应的哈密尔顿量
Pauli_h, Pauli_J, Pauli_sum = problem_PauliOperator(h, J)

# 初始化量子虚拟机, 分配量子比特
simulator = Aer.get_backend('aer_simulator')
qc = QuantumCircuit(num_qubits)

# 配置待优化参数
beta = []
gamma = []
para_list = []
for i in range(layers):
    name = "β%d" % i
    beta.append(Parameter(name))
    name = "γ%d" % i
    gamma.append(Parameter(name))
para_list = beta + gamma

# 构建QAOA
qc.append(insert_H(), [i for i in range(0, num_qubits)])
for i in range(layers):
    qc.append(oneCircuit(h, J, beta[i], gamma[i]), [i for i in range(0, num_qubits)])
qc.save_statevector()
print('Circuit Initialization Complete! Start Training...')

# 计算loss
expectation = get_expectation(qc, para_list, Pauli_sum.to_matrix(massive=True))

# 优化参数
start = time.time()
res = minimize(expectation,
               np.random.uniform(rand_start, rand_end, size=layers * 2),
               method='COBYLA',
               options={'maxiter': maxiter, 'catol': 0.000002})
print('\nTraining Done! The output of optimizer: ')
print(res)
solution = res.x
print("\nTraining done! Total elapsed time:{:.2f}s".format(time.time()-start))

# 打印结果
print_result(qc, Pauli_sum.to_matrix(), para_list, solution)