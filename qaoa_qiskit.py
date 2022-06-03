import argparse
import time
from qiskit import QuantumCircuit, transpile
from qiskit import Aer
from qiskit.algorithms.optimizers import SPSA, COBYLA, ADAM, AQGD
from qiskit.circuit import Parameter
from qiskit.providers.aer import StatevectorSimulator
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
        # statevector = np.array(_statevector)
        # statevector_dagger = np.array(_statevector.conjugate())
        # loss =  statevector_dagger @ Hamiltonian @ statevector
        loss = _statevector.expectation_value(Hamiltonian)
        assert np.imag(loss) < 1e-10
        return np.real(loss)

    return execute_circ

def str_to_statevector(string):
    string = string[::-1]
    dec = int(string, 2)
    state = np.zeros(2 ** len(string))
    state[dec] = 1.0
    return state[None,:]

def print_config():
    print('%%%%%%%%%%%%%%%%%%%% Configuration %%%%%%%%%%%%%%%%%%%%')
    print('budget: %d, g: %d, theta3: %f, layers: %d' % (budget, num_slices, theta3, layers))

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
    print("\nOptimal: selection {}, value {:.8f}".format(result[min_index][0][::-1], value_mm[min_index]))

    print("\n----------------- Full result ---------------------")
    print("rank\tselection\tvalue\t\tprobability")
    print("---------------------------------------------------")
    for i in range(len(result)):
        x, probability = result[i]
        value = value_mm[i]
        assert np.imag(value) < 1e-10
        value = np.real(value)
        # value = portfolio.to_quadratic_program().objective.evaluate(x)
        print("%d\t%-10s\t%.8f\t\t%.8f" % (i, x[::-1], value, probability))

class callback:
    def __init__(self, step_size: int):
        self.step_size = step_size
        self.full_values = []
        self._values = []
        self.values = []

    def __call__(self, nfev, parameters, value, stepsize, accepted):
        self.full_values.append(value)
        self._values.append(value)
        if len(self._values) == self.step_size:
            last_value = self._values[-1]
            self.values.append(last_value)
            self._values = []
            return self.values

def print_loss(res):
    print('%%%%%%%%%%%%%%%%%%%% Optimization Output %%%%%%%%%%%%%%%%%%%%')
    loss_ls = callback_func.values
    print('minimal loss: %s, \nmaxIter: %d' % (res[1], len(callback_func.full_values)))
    print("Parameters Found:", res[0])
    print("\n----------------- Loss (%d steps from %d iterations) -----------------" % (len(loss_ls), len(callback_func.full_values)))
    print("iter\t\tloss")
    print("------------------------------------------------------------------------")
    for i in range(len(loss_ls)):
        loss = loss_ls[i]
        # value = portfolio.to_quadratic_program().objective.evaluate(x)
        print("%d\t\t%.10f" % (i, loss))


if __name__ == '__main__':
    # 初始化参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--budget', type=int, default=3, help='Total assets.')
    parser.add_argument('--num_assets', type=int, default=6, help='The number of assets.')
    parser.add_argument('--g', type=int, default=1, help='The number of binary bits required to represent one asset.')
    parser.add_argument('--theta1', type=float, default=1.0, help='Coefficient of the linear term.')
    parser.add_argument('--theta2', type=float, default=2.5, help='Coefficient of the quadratic term.')
    parser.add_argument('--theta3', type=float, default=1.0, help='Coefficient of the Lagrangian term.')
    parser.add_argument('--seed', type=int, default=123456, help='Randon seed.')
    parser.add_argument('--optimizer', action='store_true', default=False, help='use scipy optimizer.')
    parser.add_argument('--maxiter', type=int, default=50000, help='max iterations.')
    parser.add_argument('--Gf', type=float, default=1.0, help='Granularity.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--layers', type=int, default=3, help='The number of QAOA layers.')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train.')
    args = parser.parse_args()

    # 初始化参数
    budget = args.budget
    Gf = 1.0 / budget
    theta1 = Gf
    theta2 = 2.5 * Gf * Gf
    theta3 = args.theta3
    num_assets = args.num_assets
    num_slices = args.g  # The number of binary bits required to represent one asset (g in the paper)
    layers = args.layers

    print_config()

    num_qubits = num_assets * num_slices

    np.random.seed(args.seed)
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
    expectation = get_expectation(qc, para_list, Pauli_sum)

    # 优化参数
    start = time.time()
    if args.optimizer:
        # 利用外部优化器
        res = minimize(expectation,
                       np.random.uniform(0, np.pi, size=layers * 2),
                       method='COBYLA',
                       options={'maxiter': args.maxiter})
        print('\nTraining Done! The output of optimizer: ')
        print(res)
        solution = res.x
    else:
        # 利用qiskit自带优化器
        # optimizer = COBYLA(maxiter=args.maxiter, tol=0.0001)
        # res = optimizer.optimize(num_vars=layers * 2, objective_function=expectation, initial_point=np.random.uniform(0, np.pi, size=layers * 2))
        step_size = 1 # 每隔step_size个iterations打印一次loss
        callback_func = callback(step_size)
        optimizer = SPSA(maxiter=100, blocking=True, second_order=False, callback=callback_func)
        res = optimizer.optimize(num_vars=layers * 2, objective_function=expectation, initial_point=np.random.uniform(0, np.pi, size=layers * 2))
        solution = res[0]
        # 打印loss的变化
        print_loss(res)

    print("\nTraining done! Total elapsed time:{:.2f}s".format(time.time()-start))

    # 打印结果
    print_result(qc, Pauli_sum.to_matrix(), para_list, solution)
