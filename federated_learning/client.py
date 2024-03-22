# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet3d.utils import replace_ceph_backend

import torch

import socket


def parse_args():
    parser = argparse.ArgumentParser(description='Federated Learning Client')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--server-addr', help='server address')
    parser.add_argument('--server-port', type=int, help='server-port')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    MAX_ROUNDS = 5

    rounds = 1
    global_model_path = './work_dirs/pgd_r101_fpn-head_dcn_16xb3_waymoD5-fov-mono3d_fl-s/epoch_1.pth'

    while rounds <= MAX_ROUNDS:
        # load config
        cfg = Config.fromfile(args.config)

        cfg.launcher = args.launcher

        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0], 'round_' + str(rounds))

        if rounds != 1:
            cfg.load_from = global_model_path

        # build the runner from config
        if 'runner_type' not in cfg:
            # build the default runner
            runner = Runner.from_cfg(cfg)
        else:
            # build customized runner from the registry
            # if 'runner_type' is set in the cfg
            runner = RUNNERS.build(cfg)

        # start training
        runner.train()

        # connect to server
        SERVER_ADDRESS = args.server_addr
        SERVER_PORT = args.server_port
        s = socket.socket()
        s.connect((SERVER_ADDRESS, SERVER_PORT))

        # get the path of last checkpoint and send it to the server
        file_path = cfg.work_dir + '/last_checkpoint'
        print(file_path)
        ckpt_path = open(file_path, 'r').read()
        print(ckpt_path)
        s.sendall(ckpt_path.encode())

        # wait for server to send back path of new global model
        global_model_path = s.recv(1024).decode()
        print(global_model_path)

        s.close()

        rounds += 1


if __name__ == '__main__':
    main()