import time
from pyqpanda import *
import numpy as np
from portfolio_optimization import data_preprocessing
import argparse

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
    vqc=VariationalQuantumCircuit()

    for j in qlist:
        vqc.insert(VariationalQuantumGate_RX(j,2.0*beta))

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
            vqc.insert(VariationalQuantumGate_RZ(tmp_vec[0], 2 * coef * gamma))
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
            vqc.insert(VariationalQuantumGate_CNOT(tmp_vec[0], tmp_vec[1]))
            vqc.insert(VariationalQuantumGate_RZ(tmp_vec[1], 2 * gamma * coef))
            vqc.insert(VariationalQuantumGate_CNOT(tmp_vec[0], tmp_vec[1]))
        else:
            raise AssertionError()

    return vqc

def test_coef(J, h):
    J_true = (theta2 * cov_mat + theta3 * budget ** 2 * Gf ** 2) / 4 * 2
    h_true = (theta1 * exp_ret) / 2 + theta3 * budget ** 2 * Gf * (1 - num_assets * Gf / 2) - theta2 / 4 * (
                np.sum(cov_mat, axis=0) + np.sum(cov_mat, axis=1))
    print(J == J_true)
    print(h == h_true)

def print_result(result, shoots):
    print("\n----------------- Full result ---------------------")
    print("rank\tselection\tvalue\t\tprobability")
    print("---------------------------------------------------")
    for i in range(len(result)):
        x, freq = result[i]
        value = 0.0
        # value = portfolio.to_quadratic_program().objective.evaluate(x)
        probability = freq / shoots
        print("%d\t%-10s\t%.8f\t\t%.8f" % (i, x, value, probability))

if __name__ == '__main__':
    # 初始化参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--budget', type=int, default=5, help='Total assets.')
    parser.add_argument('--num_assets', type=int, default=6, help='The number of assets.')
    parser.add_argument('--g', type=int, default=2, help='The number of binary bits required to represent one asset.')
    parser.add_argument('--theta1', type=float, default=1.0, help='Coefficient of the linear term.')
    parser.add_argument('--theta2', type=float, default=2.5, help='Coefficient of the quadratic term.')
    parser.add_argument('--theta3', type=float, default=1.0, help='Coefficient of the Lagrangian term.')
    parser.add_argument('--Gf', type=float, default=1.0, help='Granularity.')
    parser.add_argument('--layers', type=float, default=2, help='The number of QAOA layers.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Initial momentum of SGD.')
    parser.add_argument('--visual', action='store_true', default=False, help='Print the Pauli Operator of the problem.')
    parser.add_argument('--patience', type=int, default=20, help='Stop training if loss does not decrease significantly within patience steps.')
    parser.add_argument('--data_path', type=str, default="./data/stock_data.xlsx", help='The path where the original data is stored.')
    args = parser.parse_args()

    theta1 = args.theta1
    theta2 = args.theta2
    theta3 = args.theta3
    budget = args.budget
    num_assets = args.num_assets
    num_slices = args.g  # The number of binary bits required to represent one asset (g in the paper)
    Gf = args.Gf
    layers = args.layers
    epochs = args.epochs

    # 读取收益和方差
    file_path = args.data_path
    exp_ret, cov_mat = data_preprocessing(file_path)
    exp_ret = exp_ret.to_numpy()
    cov_mat = cov_mat.to_numpy()

    # 初始化量子虚拟机, 分配量子比特
    num_qubits = num_assets * num_slices
    machine = init_quantum_machine(QMachineType.CPU)

    qlist = machine.qAlloc_many(num_qubits)

    beta = var(np.ones((layers, 1), dtype='float64'), True)
    gamma = var(np.ones((layers, 1), dtype='float64'), True)

    vqc = VariationalQuantumCircuit()

    # 初始哈密尔顿量
    for i in qlist:
        vqc.insert(VariationalQuantumGate_H(i))

    # 计算所给问题对应的哈密尔顿量的系数
    J = calc_J()
    h = calc_h()
    # test_coef(J, h)
    # 计算所给问题对应的哈密尔顿量
    Hp = problem_PauliOperator(h, J)
    # 是否打印哈密尔顿量
    if args.visual:
        print(Hp)
    # 插入给定层数的QAOA layer
    if layers == 1:
        vqc.insert(oneCircuit(qlist, Hp.toHamiltonian(1), beta, gamma))
    else:
        for layer in range(layers):
            vqc.insert(oneCircuit(qlist, Hp.toHamiltonian(1), beta[layer], gamma[layer]))

    print('\nCircuit Initialization Complete! Start Training...')
    # 计算loss, 并指定优化器
    loss = qop(vqc, Hp, machine, qlist)
    optimizer = AdamOptimizer.minimize(loss,  # 损失函数
                                       args.lr,  # 学习率
                                       0.9,   # 一阶动量衰减系数
                                       0.999, # 二阶动量衰减系数
                                       1.e-10)# 很小的数值以避免零分母

    leaves = optimizer.get_variables()

    loss_value_his = 0
    loss_value_min = 1e6
    count = 0
    start = time.time()
    # 开始训练
    for i in range(epochs):
        if count > args.patience:
            break # 当连续patience个epoch输出的loss没有太大变化, 提前停止训练
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
    print("\nTraining done! Total elapsed time:{:.2f}s".format(time.time()-start))

    # 打印结果, 输出是通过measure得到的二进制字符串, 以及其出现的概率
    prog = QProg()
    qcir = vqc.feed()
    prog.insert(qcir)
    directly_run(prog)

    shoots = int(1e6)  # measure次数
    result = quick_measure(qlist, shoots)
    result_sorted = sorted(result.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)
    print_result(result_sorted, shoots)
