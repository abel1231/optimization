from qiskit import QuantumCircuit, transpile
from qiskit import Aer
from qiskit.circuit import Parameter
from portfolio_optimization import data_preprocessing
import numpy as np
from qiskit.opflow import PauliSumOp
from scipy.optimize import minimize

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


def print_result(circuit, para_list, solution):
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
    result = sorted(statevector.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    print("\n----------------- Full result ---------------------")
    print("rank\tselection\tvalue\t\tprobability")
    print("---------------------------------------------------")
    for i in range(len(result)):
        x, amplitude = result[i]
        value = 0.0
        # value = portfolio.to_quadratic_program().objective.evaluate(x)
        probability = np.abs(amplitude) ** 2
        print("%d\t%-10s\t%.8f\t\t%.8f" % (i, x, value, probability))


# 初始化参数
theta1 = 1
theta2 = 2.5
theta3 = 1
budget = 3
num_assets = 6
num_slices = 1  # The number of binary bits required to represent one asset (g in the paper)
Gf = 1.0 / 3
layers = 10

num_qubits = num_assets * num_slices

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
print('\nCircuit Initialization Complete! Start Training...')

# 计算loss
expectation = get_expectation(qc, para_list, Pauli_sum.to_matrix())

# 优化参数
res = minimize(expectation,
               np.ones(layers * 2),
               method='COBYLA')
print(res)

# 打印结果
solution = res.x
print_result(qc, para_list, solution)
