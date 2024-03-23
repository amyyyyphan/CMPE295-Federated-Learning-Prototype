import argparse
import socket
import torch


def average_weights(client_weights_list):
    num_clients = len(client_weights_list)
    avg_dict = {}

    for k in client_weights_list[0].keys():
        weights_sum = sum([state_dict[k] for state_dict in client_weights_list])
        avg_dict[k] = weights_sum / num_clients
    return avg_dict


def write_model_weights_to_file(weights, filename):
    with open(filename, 'w') as file:
        for key, value in weights.items():
            file.write(f'{key}: {value}\n')


def main():
    s = socket.socket()
    HOST = socket.gethostname()
    print('Server address: ', HOST)
    PORT = 10002
    s.bind((HOST, PORT))

    MAX_CLIENTS = 2
    MAX_ROUNDS = 5

    rounds = 1
    global_model_path = './work_dirs/pgd_r101_fpn-head_dcn_16xb3_waymoD5-fov-mono3d_fl-s/epoch_1.pth'

    client_list = []
    client_weights = []

    s.listen(MAX_CLIENTS)

    while rounds <= MAX_ROUNDS:
        try:
            conn, addr = s.accept()
            print('Connected to client: ', addr)

            client_list.append(conn)

            ckpt_path = conn.recv(1024).decode()        # client sends the path of their checkpoint
            print('Received: ' + ckpt_path)

            map_location = 'cuda:0'

            # load checkpoint
            ckpt = torch.load(ckpt_path, map_location=map_location)

            client_weights.append(ckpt['state_dict'])

            if len(client_list) == MAX_CLIENTS:
                # aggregate checkpoints weights
                global_weights = average_weights(client_weights)

                global_model = torch.load(global_model_path)

                # edit state_dict of old model and save it
                global_model['state_dict'] = global_weights
                new_model_path = './new_global_weights_' + str(rounds) + '.pth'
                torch.save(global_model, new_model_path)

                global_model_path = new_model_path
                rounds += 1

                # send path of the new global model to clients
                for client_socket in client_list:
                    client_socket.send(new_model_path.encode())
                    client_socket.close()

                client_list.clear()
                client_weights.clear()
            
        except KeyboardInterrupt:
            s.close()
            break
        except:
            s.close()
            break

    s.close()


if __name__ == '__main__':
    main()