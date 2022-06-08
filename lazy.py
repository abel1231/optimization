import time
from pyqpanda import *
import numpy as np
from portfolio_optimization import data_preprocessing
import argparse
from qiskit.algorithms.optimizers import SPSA, COBYLA, ADAM, AQGD
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

def problem_PauliOperator(h, J):
    '''
    Calculate the Pauli operator for given coefficients h and J
    :param h: coefficients of one-body terms
    :param J: coefficients of two-body terms
    :return: a PauliOperator containing the Pauli operators and its corresponding coefficients
    '''
    problem = {} # a dict containing the Pauli operator and its corresponding coefficient, such as {"Z0 Z1": 2.7, 'Z2': 1.6}
    for i in range(num_qubits):
        Pauli = 'Z' + str(i)
        problem[Pauli] = h[i]

    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            Pauli = 'Z{:d} Z{:d}'.format(i, j)
            problem[Pauli] = J[i][j]

    return PauliOperator(problem)

def oneCircuit(qlist, Hamiltonian, beta, gamma):
    vqc = QCircuit()

    for j in qlist:
        vqc.insert(RX(j,2.0*beta))

    z_dict = []
    zz_dict = []

    for i in range(len(Hamiltonian)):
        tmp_vec = []
        item = Hamiltonian[i]
        dict_p = item[0]
        for iter in dict_p:
            if 'Z' != dict_p[iter]:
                pass
            tmp_vec.append(qlist[iter])

        if 1 == len(tmp_vec):
            z_dict.append(Hamiltonian[i])
        elif 2 == len(tmp_vec):
            zz_dict.append(Hamiltonian[i])
        else:
            raise AssertionError()

    for i in range(len(z_dict)):
        tmp_vec=[]
        item=z_dict[i]
        dict_p = item[0]
        for iter in dict_p:
            if 'Z'!= dict_p[iter]:
                pass
            tmp_vec.append(qlist[iter])

        coef = item[1]

        if 1 == len(tmp_vec):
            vqc.insert(RZ(tmp_vec[0], 2 * coef * gamma))
        else:
            raise AssertionError()

    for i in range(len(zz_dict)):
        tmp_vec = []
        item = zz_dict[i]
        dict_p = item[0]
        for iter in dict_p:
            if 'Z' != dict_p[iter]:
                pass
            tmp_vec.append(qlist[iter])

        coef = item[1]

        if 2 == len(tmp_vec):
            vqc.insert(CNOT(tmp_vec[0], tmp_vec[1]))
            vqc.insert(RZ(tmp_vec[1], 2 * gamma * coef))
            vqc.insert(CNOT(tmp_vec[0], tmp_vec[1]))
        else:
            raise AssertionError()

    return vqc

def test_coef(J, h):
    J_true = (theta2 * cov_mat + theta3 * budget ** 2 * Gf ** 2) / 4 * 2
    h_true = (theta1 * exp_ret) / 2 + theta3 * budget ** 2 * Gf * (1 - num_assets * Gf / 2) - theta2 / 4 * (
                np.sum(cov_mat, axis=0) + np.sum(cov_mat, axis=1))
    print(J == J_true)
    print(h == h_true)

def stepLR(lr, cur_epoch, step_size, decay=0.99):
    if cur_epoch % step_size == 0:
        lr = lr * 0.99
    return lr

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

def qiskit_problem_PauliOperator(h, J):
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

def str_to_statevector(string):
    string = string[::-1]
    dec = int(string, 2)
    state = np.zeros(2 ** len(string))
    state[dec] = 1.0
    return state[None,:]

def print_config():
    print('%%%%%%%%%%%%%%%%%%%% Configuration %%%%%%%%%%%%%%%%%%%%')
    print('budget: %d, g: %d, theta3: %f, layers: %d' % (budget, num_slices, theta3, layers))

def get_expectation(Hamiltonian, Hamiltonian_matrix, train=True):

    def execute_circ(theta):
        p = len(theta) // 2
        beta = theta[:p]
        gamma = theta[p:]

        # beta = var(_beta.reshape(-1,1))
        # gamma = var(_gamma.reshape(-1,1))

        vqc = QCircuit()

        # 初始哈密尔顿量
        for i in qlist:
            vqc.insert(H(i))

        # 插入给定层数的QAOA layer
        if layers == 1:
            vqc.insert(oneCircuit(qlist, Hamiltonian, beta[0], gamma[0]))
        else:
            for layer in range(layers):
                vqc.insert(oneCircuit(qlist, Hamiltonian, beta[layer], gamma[layer]))

        # 构建量子线路实例
        prog = QProg()
        prog.insert(vqc)

        if train:
            # 输出每个selection对应的probability, 例如: result = {'00': 0.5, '01': 0.0, '10': 0.5, '11': 0.0}
            result = prob_run_list(prog, qlist, -1)
            statevector = np.sqrt(np.array(result))  # vector representation of the output state

            loss = statevector @ Hamiltonian_matrix @ statevector
            assert np.imag(loss) < 1e-10
            return np.real(loss)
        else:
            result = prob_run_dict(prog, qlist, -1)
            result_tonumpy = np.array(prob_run_list(prog, qlist, -1))
            return result, result_tonumpy
    return execute_circ

def print_result(Hamiltonian, Hamiltonian_matrix, solution):
    execute_circ = get_expectation(Hamiltonian, Hamiltonian_matrix, train=False)
    result, result_tonumpy = execute_circ(solution)

    result = sorted(result.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    mm = []
    for i in range(len(result)):
        x, _ = result[i]
        mm.append(str_to_statevector(x))
    mm = np.concatenate(mm, axis=0)
    value_mm = np.sum((mm @ Hamiltonian_matrix) * mm, axis=1)

    min_index = np.argmin(value_mm)

    print("\nOptimal: selection {}, value {:.8f}".format(result[min_index][0][::-1], value_mm[min_index]))

    print("\n----------------- Full result ---------------------")
    print("rank\tselection\tvalue\t\tprobability")
    print("---------------------------------------------------")
    value_save = []
    probability_save = []
    utility_save = []
    for i in range(len(result)):
        x, probability = result[i]
        value = value_mm[i]
        assert np.imag(value) < 1e-10
        value = np.real(value)
        # value = portfolio.to_quadratic_program().objective.evaluate(x)
        print("%d\t%-10s\t%.8f\t\t%.8f" % (i, x[::-1], value, probability))
        ## do not save the optimal selection
        # np.savez("./output/budget_{}_layers_{}_theta3_{}.npz".format(budget, layers, theta3),
        #          value=np.array(value_save), \
        #          probability=np.array(probability_save), utility=np.array(utility_save))

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
    print('minimal loss: %s, \nmaxIter: %d, func_eval: %d' % (res[1], len(callback_func.full_values), res[2]))
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
    parser.add_argument('--Gf', type=float, default=1.0, help='Granularity.')
    parser.add_argument('--optimizer', action='store_true', default=False, help='use scipy optimizer.')
    parser.add_argument('--maxiter', type=int, default=300, help='max iterations.')
    parser.add_argument('--layers', type=float, default=3, help='The number of QAOA layers.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--visual', action='store_true', default=False, help='Print the Pauli Operator of the problem.')
    parser.add_argument('--data_path', type=str, default="./data/stock_data.xlsx", help='The path where the original data is stored.')
    args = parser.parse_args()

    budget = args.budget
    Gf = 1.0 / budget
    theta1 = Gf
    theta2 = 2.5 * Gf * Gf
    theta3 = args.theta3
    num_assets = args.num_assets
    num_slices = args.g  # The number of binary bits required to represent one asset (g in the paper)
    layers = args.layers
    maxiter = args.maxiter
    optimizer = args.optimizer

    print_config()

    # 读取收益和方差
    file_path = args.data_path
    exp_ret, cov_mat = data_preprocessing(file_path)
    exp_ret = exp_ret.to_numpy()
    cov_mat = cov_mat.to_numpy()

    # 初始化量子虚拟机, 分配量子比特
    num_qubits = num_assets * num_slices
    machine = init_quantum_machine(QMachineType.CPU)

    qlist = machine.qAlloc_many(num_qubits)

    # 计算所给问题对应的哈密尔顿量的系数
    J = calc_J()
    h = calc_h()
    # test_coef(J, h)
    # 计算所给问题对应的哈密尔顿量, 及其对应的矩阵
    Hp = problem_PauliOperator(h, J)

    Pauli_h, Pauli_J, Pauli_sum = qiskit_problem_PauliOperator(h, J)
    Hamiltonian_matrix = Pauli_sum.to_matrix()
    # 是否打印哈密尔顿量
    if args.visual:
        print(Hp)

    # print('\nCircuit Initialization Complete! Start Training...')

    # 计算loss
    expectation = get_expectation(Hp.toHamiltonian(1), Hamiltonian_matrix)

    # 优化参数
    start = time.time()
    if optimizer:
        # 利用外部优化器
        res = minimize(expectation,
                       np.random.uniform(-0.1, 0.1, size=layers * 2),
                       method='COBYLA',
                       options={'maxiter': args.maxiter})
        print('\nTraining Done! The output of optimizer: ')
        print(res)
        solution = res.x
    else:
        # 利用qiskit自带优化器
        # optimizer = COBYLA(maxiter=args.maxiter, tol=0.0001)
        # res = optimizer.optimize(num_vars=layers * 2, objective_function=expectation, initial_point=np.random.uniform(0, np.pi, size=layers * 2))
        step_size = 1  # 每隔step_size个iterations打印一次loss
        callback_func = callback(step_size)
        optimizer = SPSA(maxiter=maxiter, blocking=True, second_order=True, callback=callback_func)
        res = optimizer.optimize(num_vars=layers * 2, objective_function=expectation,
                                 initial_point=np.random.uniform(-0.1, 0.1, size=layers * 2))
        solution = res[0]
        # 打印loss的变化
        print_loss(res)

    print("\nTraining done! Total elapsed time:{:.2f}s".format(time.time() - start))

    # 打印结果
    print_result(Hp.toHamiltonian(1), Pauli_sum.to_matrix(), solution)
