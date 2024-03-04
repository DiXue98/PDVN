import matplotlib.pyplot as plt

import pathlib
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument('--result_folder', type=str)

# def diff(data):
#     for i in range(len(data)-1, 0, -1):
#         if data[i] > data[i-1]:
#             data[i] -= data[i-1]
#     return data

def culmulated_sum(data):
    for i in range(1, len(data)):
        data[i] += data[i-1]
    return data

def second2hour(data):
    for i in range(len(data)):
        data[i] /= 3600
    return data

def sum_each_mol(*args):
    tot = []
    for i in range(len(args[0])):        
        tot.append(0)
        for data in args:
            tot[i] += data[i]
    return tot

if __name__ == '__main__':
    args = parser.parse_args()
    result_folder = pathlib.Path(args.result_folder)
    timer_file = result_folder.joinpath('timer.json')
    f = open(timer_file)
    timer = json.load(f)

    num_mols = timer['num_mols']
    rollout = timer['rollout']
    extra_lesson = timer['extra_lesson']
    training = timer['training']
    # for i in range(len(rollout)):
    #     if rollout[i] > 2000:
    #         print(i)
    # culmulated_sum(diff(num_mols))
    # culmulated_sum(num_mols)
    second2hour(culmulated_sum(rollout))
    second2hour(culmulated_sum(extra_lesson))
    second2hour(culmulated_sum(training))
    tot = sum_each_mol(rollout, extra_lesson, training)

    fig, ax = plt.subplots()
    ax.plot(num_mols, rollout, label='rollout')
    ax.plot(num_mols, extra_lesson, label='extra_lesson')
    ax.plot(num_mols, training, label='training')
    ax.plot(num_mols, tot, label='total')
    ax.set_xlabel('# of mols')
    ax.set_ylabel('time/h')

    plt.legend(loc='best')
    plt.savefig('timer.png')