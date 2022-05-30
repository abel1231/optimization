import time
from pyqpanda import *
import numpy as np
from portfolio_optimization import data_preprocessing
import argparse

def calc_J():
    '''
    calculate the coefficients of all rzz gates
    :return: a (num_assets*num_assets*num_slices*num_slices) coefficient matrix
    '''
    coef_J = 2 ** (num_slices-3)  * np.ones((num_slices, num_slices))
    for i in range(1, num_slices):
        diag = np.flipud(np.diag(np.power(2, i) * np.ones(num_slices - i), k=i))
        coef_J = coef_J * np.where(diag != 0, diag, 1.0)
        diag = np.flipud(np.diag(1.0 / np.power(2, i) * np.ones(num_slices - i), k=-i))
        coef_J = coef_J * np.where(diag != 0, diag, 1.0)

    J = (theta3 * budget**2 * Gf**2 + theta2 * cov_mat[:, :, None, None]) * coef_J[None, None, :]
    return J

def calc_h(J):
    '''
    calculate the coefficients of all rz gates
    :return: a (num_assets*num_slices) coefficient matrix
    '''
    h = np.zeros((num_assets, num_slices))
    for i in range(num_assets):
        for k in range(num_slices):
            h[i][k] = 2**(k-1) * (-theta1 * exp_ret[i] - 2*theta3 * budget**2 * Gf)
    h = h + np.sum(J, axis=(1,3))
    return h

def problem_PauliOperator(h, J):
    '''
    Calculate the Pauli operator for  given coefficients h and J
    :param h: coefficients of one-body terms
    :param J: coefficients of two-body terms
    :return: a PauliOperator containing the Pauli operator and its corresponding coefficient
    '''
    PauliOp = {} # a dict containing the Pauli operator and its corresponding coefficient, such as {"Z0 Z1": 2.7, 'Z2': 1.6}
    for i in range(num_assets):
        for k in range(num_slices):
            Pauli = 'Z' + str(i * num_slices + k)
            PauliOp[Pauli] = h[i][k]

    if num_slices == 1:
        calc_list = [(i, j) for i in range(num_assets) for j in range(i + 1, num_assets)]
    else:
        calc_list = [(i, j) for i in range(num_assets) for j in range(i, num_assets)]

    for i, j in calc_list:
        if i < j:
            for k1 in range(num_slices):
                for k2 in range(num_slices):
                    Pauli = 'Z{:d} Z{:d}'.format(i * num_slices + k1, j * num_slices + k2)
                    PauliOp[Pauli] = J[i][j][k1][k2]
        elif i == j:
            for k1 in range(num_slices):
                for k2 in range(k1+1, num_slices):
                    Pauli = 'Z{:d} Z{:d}'.format(i * num_slices + k1, j * num_slices + k2)
                    PauliOp[Pauli] = J[i][j][k1][k2]
        else:
            raise AssertionError()

    return PauliOperator(PauliOp)

def oneCircuit(qlist, Hamiltonian, beta, gamma):
    '''
    construct the circuit corresponding to one QAOA layer
    :param qlist: dictionary containing qubits assigned to each asset
    :param h: coefficients of all rz gates
    :param J: coefficients of all rzz gates
    :param beta: parameter of the mixer term
    :param gamma: parameter of the problem Hamiltonian term
    :return: the constructed circuit
    '''
    vqc = VariationalQuantumCircuit()

    assert len(qlist) == num_assets * num_slices

    for i in range(len(Hamiltonian)):
        tmp_vec = []
        flag = 'z'
        item = Hamiltonian[i]
        dict_p = item[0]
        for iter in dict_p:
            if 'Z' != dict_p[iter]:
                pass
            else:
                tmp_vec.append(qlist[iter])
                flag = 'z'

        coef = item[1]

        # e^(-iγH)
        if len(tmp_vec) == 1:
            vqc.insert(VariationalQuantumGate_RZ(tmp_vec[0], 2 * coef * gamma))
        elif len(tmp_vec) == 2:
            vqc.insert(VariationalQuantumGate_CNOT(tmp_vec[0], tmp_vec[1]))
            vqc.insert(VariationalQuantumGate_RZ(tmp_vec[1], 2 * gamma * coef))
            vqc.insert(VariationalQuantumGate_CNOT(tmp_vec[0], tmp_vec[1]))
        else:
            raise AssertionError()

    # e^(-iβσx)
    for i in qlist:
        vqc.insert(VariationalQuantumGate_RX(i, 2.0 * beta))
    return vqc


if __name__ == '__main__':
    # 初始化参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--budget', type=int, default=1, help='Total assets.')
    parser.add_argument('--num_assets', type=int, default=6, help='The number of assets.')
    parser.add_argument('--g', type=int, default=1, help='The number of binary bits required to represent one asset.')
    parser.add_argument('--theta3', type=float, default=1.0, help='Coefficient of Lagrangian term.')
    parser.add_argument('--Gf', type=float, default=1.0, help='Granularity.')
    parser.add_argument('--layers', type=float, default=1.0, help='The Number of QAOA layers.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--momentum', type=float, default=0.01, help='Initial momentum of SGD.')
    parser.add_argument('--visual', action='store_true', default=True, help='Print the Pauli Operator of the problem.')
    parser.add_argument('--data_path', type=str, default="./data/stock_data.xlsx", help='The path where the original data is stored.')
    args = parser.parse_args()

    theta1 = 1
    theta2 = 2.5
    theta3 = args.theta3
    budget = args.budget
    num_assets = args.num_assets
    num_slices = args.g  # the number of slices of the budget (g in the paper)
    Gf = args.Gf
    layers = args.layers
    epochs = args.epochs

    # 读取收益和方差
    file_path = args.data_path
    exp_ret, cov_mat = data_preprocessing(file_path)
    exp_ret = exp_ret.to_numpy()
    cov_mat = cov_mat.to_numpy()

    # 初始化量子虚拟机，分配量子比特
    num_qubit = num_assets * num_slices
    machine = init_quantum_machine(QMachineType.CPU)

    qlist = machine.qAlloc_many(num_qubit)

    beta = var(np.ones((layers, 1), dtype='float64'), True)
    gamma = var(np.ones((layers, 1), dtype='float64'), True)

    vqc = VariationalQuantumCircuit()

    # 初始哈密尔顿量
    for i in qlist:
            vqc.insert(VariationalQuantumGate_H(i))

    # 计算所给问题对应的哈密尔顿量的系数
    J = calc_J()
    h = calc_h(J)
    # 计算所给问题对应的哈密尔顿量
    Hp = problem_PauliOperator(h, J)
    # 是否打印哈密尔顿量
    if args.visual:
        print(Hp)
    # 插入给定层数的QAOA layer
    for layer in range(layers):
        vqc.insert(oneCircuit(qlist, Hp.toHamiltonian(1), beta[layer], gamma[layer]))

    # 计算loss，并指定优化器
    loss = qop(vqc, Hp, machine, qlist)
    optimizer = MomentumOptimizer.minimize(loss, args.lr, args.momentum)

    leaves = optimizer.get_variables()

    loss_value_his = 0
    loss_value_min = 0
    count = 0
    start = time.time()
    # 开始训练
    for i in range(epochs):
        # 当连续20个epoch输出的loss没有太大变化，提前停止训练
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

    # 打印结果，输出是通过measure得到的二进制字符串，以及其出现的次数
    prog = QProg()
    qcir = vqc.feed()
    prog.insert(qcir)
    directly_run(prog)

    result = quick_measure(qlist, int(1e6))
    print(sorted(result.items(), key = lambda kv:(kv[1], kv[0]), reverse=True))


def test_coef_J(J):
    t = np.zeros((num_assets, num_assets, num_slices, num_slices))
    for i in range(num_assets):
        for j in range(num_assets):
            for m in range(0, num_slices):
                for n in range(0, num_slices):
                    t[i][j][m][n] = (theta3 * budget**2 * Gf**2 + theta2 * cov_mat[i][j]) * 2**((m+n+2)-4)


    return (t==J).sum()
