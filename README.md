﻿# CMPE295 Federated Learning Prototype

## Instructions
1. Copy `federated_learning` into the `mmdetection3d` directory

2. Start the server
```
python federated_learning/server.py
```

3. Start a client (replace the config file path with your config file path and `--server-addr` to the gpu node that the server is running on)
```
python federated_learning/client.py configs/pgd/pgd_r101_fpn-head_dcn_16xb3_waymoD5-fov-mono3d_fl-c1.py --server-addr g15 --server-port 10002
```
