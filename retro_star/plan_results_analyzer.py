import argparse
import pathlib
import json

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--result_folder', type=str)
parser.add_argument('--plan_info', type=str)

if __name__ == '__main__':
    overall_result_w_value = []
    overall_result_wo_value = []
    args = parser.parse_args()
    path = pathlib.Path(args.result_folder)    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # without value
    for checkpoint_folder in path.iterdir():
        result_file = checkpoint_folder.joinpath(args.plan_info, 'result_iter.json')
        if result_file.exists():
            f = open(result_file)
            result = json.load(f)
            x = int(checkpoint_folder.name[:-5])
            y = np.sum(result['succ']['500'])
            z = result['avg_lens']['500']
            overall_result_wo_value.append((x, y, z))

    overall_result_wo_value = sorted(overall_result_wo_value, key=lambda x:x[0])
    print(overall_result_wo_value)
    x, y, z = zip(*overall_result_wo_value)
    ax1.plot(x, y, color='g', label='succ')
    ax2.plot(x, z, color='b', label='avg_lens')

    # ax1.set_ylim([160, 190])
    # ax2.set_ylim([6.5, 10.0])
    ax1.set_xlabel("# of mols")
    ax1.set_ylabel("# of succ")
    ax2.set_ylabel("average length")
    fig.legend(loc='upper left')
    # fig.legend(bbox_to_anchor=(0.45, 1.2))
    # fig.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
    # fig.tight_layout()
    plt.savefig(args.result_folder + f'{args.plan_info}.png')