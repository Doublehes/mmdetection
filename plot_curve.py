#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@File    :   plot_curve.py
@Time    :   2024/07/19 10:55:02
@Author  :   shuang.he
@Version :   1.0
@Contact :   shuang.he@momenta.ai
@License :   Copyright 2024, Momenta/shuang.he
@Desc    :   None
'''



from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
import os

from tools.analysis_tools.analyze_logs import load_json_logs



class Args:
    def __init__(self, json_logs, style='darkgrid', legend=None, keys=None, start_epoch=0, title="plot", out='plot.png'):
        self.style = style
        self.start_epoch = start_epoch
        self.title = title
        self.out = out
        self.json_logs = json_logs
        self.legend = ['loss_cls', 'loss_bbox', 'loss_centerness', 'loss'] if legend is None else legend
        self.keys = ['loss_cls', 'loss_bbox', 'loss_centerness', 'loss'] if keys is None else keys


def plot_curve(log_dicts: dict, args: Args):

    sns.set_style(args.style)

    legend = args.legend
    metrics = args.keys

    plt.figure(figsize=(20, 15))
    # TODO: support dynamic eval interval(e.g. RTMDet) when plotting mAP.
    num_metrics = len(metrics)
    for i, log_dict in enumerate(log_dicts):
        epochs = [e for e in log_dict.keys() if e >= args.start_epoch]
        for j, metric in enumerate(metrics):
            print(f'plot curve of {args.json_logs[i]}, metric is {metric}')

            if 'AP' in metric:
                xs = []
                ys = []
                for epoch in epochs:
                    ys += log_dict[epoch][metric]
                    if log_dict[epoch][metric]:
                        xs += [epoch]
                # print(f"metric: {metric}, len(xs): {len(xs)}, len(ys): {len(ys)}")
                if len(xs) != len(ys):
                    print(f'metric {metric} has different length of iters and values')
                    continue
                if len(xs) == 0:
                    print(f'metric {metric} has no value')
                    continue
                plt.xlabel('epoch')
                plt.plot(xs, ys, label=legend[i * num_metrics + j], marker='o')
            else:
                xs = []
                ys = []
                for epoch in epochs:
                    iters = log_dict[epoch]['step']
                    xs.append(np.array(iters))
                    ys.append(np.array(log_dict[epoch][metric][:len(iters)]))
                xs = np.concatenate(xs)
                ys = np.concatenate(ys)
                # print(f"metric: {metric}, len(xs): {len(xs)}, len(ys): {len(ys)}")
                if len(xs) != len(ys):
                    print(f'metric {metric} has different length of iters and values')
                    continue
                plt.xlabel('iter')
                plt.plot(xs, ys, label=legend[i * num_metrics + j], linewidth=0.5)
    plt.legend()
    if args.title is not None:
        plt.title(args.title)

    print(f'save curve to: {args.out}')
    plt.savefig(args.out)
    plt.cla()




if __name__ == '__main__':
    assert len(sys.argv) == 2
    json_logs = sys.argv[1:]
    for json_log in json_logs:
        assert json_log.endswith('.json')

    log_dicts = load_json_logs(json_logs)

    out_dir = os.path.dirname(os.path.dirname(json_logs[0]))

    loss_args = Args(json_logs, 
                     keys=['loss_cls', 'loss_bbox', 'loss_centerness', 'loss'],
                     legend=['loss_cls', 'loss_bbox', 'loss_centerness', 'loss'],
                     start_epoch=2,
                     title='Loss Curve',
                     out=os.path.join(out_dir, 'loss.png')
                     )
    plot_curve(log_dicts, loss_args)

    lr_args = Args(json_logs, 
                   keys=['lr'],
                   legend=['lr'],
                   title='Learning Rate Curve',
                   out=os.path.join(out_dir, 'lr.png') 
                   )
    plot_curve(log_dicts, lr_args)

    mAP_args = Args(json_logs,
                    keys=['bbox_mAP', 'mAP', 'AP50', 'AP75'],
                    legend=['bbox_mAP', 'mAP', 'AP50', 'AP75'],
                    title='mAP Curve',
                    out=os.path.join(out_dir, 'mAP.png')
                    )
    plot_curve(log_dicts, mAP_args)

    mAP_args = Args(json_logs,
                    keys=['AP50_small', 'AP50_medium', 'AP50_large'],
                    legend=['AP50_small', 'AP50_medium', 'AP50_large'],
                    title='AP Curve',
                    out=os.path.join(out_dir, 'AP.png')
                    )
    plot_curve(log_dicts, mAP_args)
