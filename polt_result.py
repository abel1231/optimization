import os.path
import matplotlib.pyplot as plt
import numpy as np
import argparse

# 初始化参数
parser = argparse.ArgumentParser()
parser.add_argument('--num_display', type=int, default=5)
parser.add_argument('--offset', type=float, default=0.0003)
parser.add_argument('--path', type=str, default='./out/b2_g1_ly8.out')
parser.add_argument('--save_path', type=str, default='./out')
parser.add_argument('--save_type', type=str, default='eps')
args = parser.parse_args()

path = args.path
offset = args.offset
num_display = args.num_display
save_type = args.save_type
assert save_type in ['png', 'pdf', 'svg', 'eps']

# 读取selection和对应的probability
with open(path, 'r', encoding="utf-8") as f:
    file = f.readlines()

anchor = file.index('----------------- Full result ---------------------\n')
optimal_value = np.array(eval(file[anchor-2].split()[-1])).real.item()

data = []
for item in file[anchor+3:]:
    item = item.split()
    value, prob = eval(item[2]), eval(item[3])
    data.append((value, prob))

data = sorted(data, key=lambda kv: kv[0])

result = {}
value_unique = data[0][0]
result[value_unique] = 0.0
for value, prob in data:
    if value - value_unique <= offset:
        result[value_unique] += prob
    else:
        value_unique = value
        result[value_unique] = prob

# debug
probsum = 0.0
for i in result.values():
    probsum = probsum + i
print('probsum: %f' % probsum)

# 只plot前num_display个energy最小的, 其他的统统表示为'others'
if len(result) > num_display:
    others_prob = sum(list(result.values())[num_display:])
    probability = list(result.values())[:num_display]
    probability.append(others_prob)
    energy = list(result.keys())[:num_display]
    labels = ['%.4f' % i for i in energy]
    labels.append('others')
else:
    energy = list(result.keys())
    probability = list(result.values())
    labels = ['%.4f' % i for i in energy]

probsum = 0.0
for i in probability:
    probsum = probsum + i
print('probsum: %f' % probsum)


# plot figure
y_pos = np.arange(len(labels))
fig, ax = plt.subplots()

hbars = ax.barh(y_pos, probability, align='center')
ax.set_yticks(y_pos, labels=labels)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Cumulative Probability')
ax.set_ylabel('Energy')
# ax.set_title('How fast do you want to go today?')

# Label with specially formatted floats
ax.bar_label(hbars, fmt='%.4f')
ax.set_xlim(right=1.0)  # adjust xlim to fit labels
plt.subplots_adjust(left=0.2)

filename = path.split('/')[-1].split('.')[0] + ('_categories_%d_offset_%.4f'%(num_display, offset)).replace('.', '-') + '.' + save_type
filename = os.path.join(args.save_path, filename)
print(filename)
plt.savefig(filename, format=save_type)

plt.show()
