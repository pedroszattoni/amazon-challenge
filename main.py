"""
Solve Amazon Challenge using Inverse Optimization.

Author: Pedro Zattoni Scroccaro
"""

import sys
import pickle
from utils import (route_to_zone_seq, solve_ATSP, zone_seq_to_vec,
                   amazon_score, zone_centers)
import numpy as np
import time
import argparse
import invopt as iop

# %%%%%%%%%%%%%%%%%%%%%%%%%%% Simulation paramters %%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Check if the script in being run through command-line or as a script in an
# IDE, e.g., Spyder
if len(sys.argv) > 1:
    parser = argparse.ArgumentParser()
    parser.add_argument('station_code',
                        choices=['DLA7', 'DLA9', 'DLA8', 'DBO3', 'DSE5',
                                 'DSE4', 'DCH4', 'DBO2', 'DCH3', 'DLA3',
                                 'DLA4', 'DAU1', 'DCH1', 'DLA5', 'DSE2',
                                 'DCH2', 'DBO1'],
                        help='Station (a.k.a. depot) code.')
    parser.add_argument('solver_IO', choices=['gurobi', 'ortools', 'greedy'],
                        help=('Solver used in the IO algorithm.'))
    parser.add_argument('solver_complete_route',
                        choices=['gurobi', 'ortools', 'greedy'],
                        help=('Solver used in the IO algorithm.'))
    parser.add_argument('step_size_constant', type=float,
                        help=('Constant that multiplies the step-size used ' +
                              'for the IO algorithm.'))
    parser.add_argument('step_size_type', choices=['1/t', '1/sqrt(t)'],
                        help=('Type of step-size.'))
    parser.add_argument('T', type=int,
                        help=('Number of iterations of the IO algorithm.'))
    parser.add_argument('resolution', type=int,
                        help=('Amazon score is computed every RESOLUTION ' +
                              'iterations of the IO algorithm.'))
    parser.add_argument('update_step', choices=['standard', 'exponentiated'],
                        help=('Type of update step.'))
    parser.add_argument('regularizer', choices=['L2_squared', 'L1'],
                        help=('Type of regularizer.'))
    parser.add_argument('reg_param', type=float,
                        help=('Nonnegative IO regularization parameter'))
    parser.add_argument('averaged_type', type=int,
                        choices=[0, 1, 2],
                        help=('Type of averaging of the iterates.'))
    parser.add_argument('path_to_input_data',
                        help=('Path to the folder where the pre-processed ' +
                              'Amazon Challenge data is located.'))
    parser.add_argument('path_to_output_data',
                        help=('Path to the folder where the script results ' +
                              'should be saved.'))
    parser.add_argument('--sub_loss', action='store_true',
                        help=('Use suboptimality loss instead of augmented.'))
    parser.add_argument('--normalize_grad', action='store_true',
                        help=('Nomalize gradients.'))
    parser.add_argument('--compute_train_score', action='store_true',
                        help=('Also compute the Amazon score for the ' +
                              'training data, i.e., the in-sample score.'))
    parser.add_argument('--area_cluster', action='store_true',
                        help=('Assumes expert respects area clusters, and ' +
                              'enforce this behaviour in the solution.'))
    parser.add_argument('--region_cluster', action='store_true',
                        help=('Assumes expert respects region clusters, and ' +
                              'enforce this behaviour in the solution.'))
    parser.add_argument('--zone_id_diff', action='store_true',
                        help=('Assumes expert respects the ' +
                              "'one unit difference' rule, and enforce " +
                              'this behaviour in the solution.'))
    args = parser.parse_args()

    station_code = args.station_code
    solver_IO = args.solver_IO
    solver_complete_route = args.solver_complete_route
    step_size_constant = args.step_size_constant
    step_size_type = args.step_size_type
    T = args.T
    resolution = args.resolution
    update_step = args.update_step
    regularizer = args.regularizer
    reg_param = args.reg_param
    averaged_type = args.averaged_type
    path_to_input_data = args.path_to_input_data
    path_to_output_data = args.path_to_output_data
    sub_loss = args.sub_loss
    normalize_grad = args.normalize_grad
    compute_train_score = args.compute_train_score
    area_cluster = args.area_cluster
    region_cluster = args.region_cluster
    zone_id_diff = args.zone_id_diff
else:
    # ['DLA7', 'DLA9', 'DLA8', 'DBO3', 'DSE5',
    #          'DSE4', 'DCH4', 'DBO2', 'DCH3', 'DLA3',
    #          'DLA4', 'DAU1', 'DCH1', 'DLA5', 'DSE2',
    #          'DCH2', 'DBO1']
    station_code = 'DBO1'
    solver_IO = 'gurobi'
    solver_complete_route = 'ortools'
    compute_train_score = True
    area_cluster = True
    region_cluster = True
    zone_id_diff = True
    step_size_constant = 0.0005
    step_size_type = '1/t'
    resolution = 1
    T = 5
    update_step = 'standard'
    regularizer = 'L2_squared'
    reg_param = 0
    averaged_type = 2
    normalize_grad = False
    sub_loss = False
    path_to_input_data = 'path/to/input/data/'
    path_to_output_data = 'path/to/output/data/'

print(f'station_code = {station_code}')
print(f'solver_IO = {solver_IO}')
print(f'solver_complete_route = {solver_complete_route}')
print(f'sub_loss = {sub_loss}')
print(f'compute_train_score = {compute_train_score}')
print(f'area_cluster = {area_cluster}')
print(f'region_cluster = {region_cluster}')
print(f'zone_id_diff = {zone_id_diff}')
print(f'step_size_constant = {step_size_constant}')
print(f'step_size_type = {step_size_type}')
print(f'resolution = {resolution}')
print(f'T = {T}')
print(f'update_step = {update_step}')
print(f'regularizer = {regularizer}')
print(f'reg_param = {reg_param}')
print(f'averaged_type = {averaged_type}')
print(f'normalize_grad = {normalize_grad}')


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Load data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print('')
print('Loading data...')
tic = time.time()
dataset_train_raw = pickle.load(open(path_to_input_data + station_code +
                                     '_train.p', "rb"))
dataset_test = pickle.load(open(path_to_input_data + station_code +
                                '_test.p', "rb"))
toc = time.time()
print(f'Done! ({round(toc-tic, 2)} seconds)')


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Process data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Filter training dataset according to route quality and/or if the route
# contains undelivered packages
rq = 'HML'
if rq == 'HM':
    route_quality = ['High', 'Medium']
elif rq == 'HML':
    route_quality = ['High', 'Medium', 'Low']
delivered = [True, False]
dataset_train = {}
for route_ID in dataset_train_raw:
    if ((dataset_train_raw[route_ID]['route_score'] in route_quality)
            and (dataset_train_raw[route_ID]['delivered'] in delivered)):
        dataset_train[route_ID] = dataset_train_raw[route_ID].copy()

# Extract zone_ids from training dataset and create dict of all stops per zone.
zone_ids_train = set()
for route_ID in dataset_train:
    stops = dataset_train[route_ID]['stops']
    for stop in stops:
        zone_id = stops[stop]['zone_id']
        zone_ids_train.add(zone_id)

m_train = len(zone_ids_train)

# Zone centers
zc_train, _ = zone_centers(dataset_train)

# Associate every zone ID (str) with a unique index (int). This makes it easier
# to work with the zones. Depot gets index=0
list_zones = list(zone_ids_train)
list_zones.insert(0, list_zones.pop(list_zones.index('depot')))
m_total = len(list_zones)
indexes = list(range(m_total))
zone_id_to_index = dict(zip(list_zones, indexes))
index_to_zone_id = dict(zip(indexes, list_zones))

# Create zone to area dict
index_to_area = {zone_id_to_index[zone]: zone[:-2]+zone[-1]
                 for zone in list_zones}

# Create zone to region dict
index_to_region = {zone_id_to_index[zone]: zone[:-3] for zone in list_zones}

# Create IO datasets of signal and expert response
dataset_train_z = []
for route_ID in dataset_train:
    zone_seq = route_to_zone_seq(dataset_train[route_ID]['stops'],
                                 zone_id_to_index)
    dataset_train_z.append(((set(zone_seq), m_train), zone_seq))


# %%%%%%%%%%%%%%%%%%%%%%%%%%% IO functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# See InvOpt package documentation for details:
# https://github.com/pedroszattoni/invopt

def step_size_func(t):
    """Step-size function."""
    if step_size_type == '1/t':
        ss = step_size_constant/(1+t)
    elif step_size_type == '1/sqrt(t)':
        ss = step_size_constant/np.sqrt(1+t)
    return ss


def phi(s, x):
    """Feature mapping."""
    _, m = s
    x_vec = zone_seq_to_vec(x, m)
    return x_vec


def FOP(theta, s):
    """Forward optimization problem."""
    zones_set, m = s

    if len(theta) != m**2:
        raise Exception('Length of theta vector does not match number of ' +
                        'zones!')

    theta_mat = theta.reshape(m, m)

    dists = {}
    for i in zones_set:
        for j in zones_set:
            if i != j:
                # Enforcer clustering of areas
                if area_cluster:
                    M_A = (index_to_area[i] != index_to_area[j])
                else:
                    M_A = 0

                # Enforcer clustering of regions
                if region_cluster:
                    M_R = (index_to_region[i] != index_to_region[j])
                else:
                    M_R = 0

                # Penalty of 1 for each token that changes by more than one
                # zone ID = W-x.yZ
                if zone_id_diff:
                    zone_id_i = index_to_zone_id[i]
                    zone_id_j = index_to_zone_id[j]

                    W_i = ord(zone_id_i[0])
                    W_j = ord(zone_id_j[0])
                    M_W = (np.abs(W_i - W_j) > 1.5)

                    try:
                        x_i = int(zone_id_i[2:-3])
                        x_j = int(zone_id_j[2:-3])
                        M_x = (np.abs(x_i - x_j) > 1.5)
                    except ValueError:
                        M_x = 1

                    try:
                        y_i = int(zone_id_i[-2])
                        y_j = int(zone_id_j[-2])
                        M_y = (np.abs(y_i - y_j) > 1.5)
                    except ValueError:
                        M_y = 1

                    Z_i = ord(zone_id_i[-1])
                    Z_j = ord(zone_id_j[-1])
                    M_Z = (np.abs(Z_i - Z_j) > 1.5)

                    M_token = M_W + M_x + M_y + M_Z
                else:
                    M_token = 0

                dists[(i, j)] = M_A + M_R + M_token + theta_mat[i, j]

    x_opt, _ = solve_ATSP(dists, solver_IO)

    return x_opt


def FOP_aug(theta, s_hat, x_hat):
    """Augmented FOP."""
    _, m = s_hat
    x_hat_vec = zone_seq_to_vec(x_hat, m)

    theta_aug = theta - (1 - 2*x_hat_vec)
    x_aug = FOP(theta_aug, s_hat)

    return x_aug


def callback(theta):
    """Store iterate."""
    return theta


# %%%%%%%%%%%%%%%%%%%%%%%%%%% IO training/testing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Compute initial theta
initial_theta = 'zone_centers'
p = m_train**2
if initial_theta == 'uniform':
    theta_0 = np.ones(p)
elif initial_theta == 'zone_centers':
    theta_0 = np.ones((m_train, m_train))
    for i in range(m_train):
        for j in range(m_train):
            zone_id_i = index_to_zone_id[i]
            zone_id_j = index_to_zone_id[j]
            dist_lat = zc_train[zone_id_i][0] - zc_train[zone_id_j][0]
            dist_lng = zc_train[zone_id_i][1] - zc_train[zone_id_j][1]
            theta_0[i][j] = np.linalg.norm([dist_lat, dist_lng])
    theta_0 = theta_0.flatten()

score_train_hist = []
score_test_hist = []

print('')
print('Inverse optimization...')
tic = time.time()
if sub_loss:
    FOP_FOM = FOP
else:
    FOP_FOM = FOP_aug
theta_IO_list = iop.FOM(dataset_train_z, phi, theta_0, FOP_FOM, step_size_func,
                        T,
                        Theta='nonnegative',
                        step=update_step,
                        regularizer=regularizer,
                        reg_param=reg_param,
                        batch_type='reshuffled',
                        averaged=averaged_type,
                        callback=callback,
                        normalize_grad=normalize_grad)
toc = time.time()
print(f'Done! ({round(toc-tic, 2)} seconds)')

print('')
print('Computing Amazon scores...')
tic = time.time()
T_list = list(range(0, T+resolution, resolution))
for T in T_list:
    theta_IO = theta_IO_list[T]

    if compute_train_score:
        scores_train = amazon_score(theta_IO, dataset_train, zc_train,
                                    zone_id_to_index, index_to_zone_id,
                                    solver_complete_route, solver_IO,
                                    station_code, area_cluster, region_cluster,
                                    zone_id_diff)
        score_train_hist.append(scores_train['submission_score'])

    scores_test = amazon_score(theta_IO, dataset_test, zc_train,
                               zone_id_to_index, index_to_zone_id,
                               solver_complete_route, solver_IO, station_code,
                               area_cluster, region_cluster, zone_id_diff)
    score_test_hist.append(scores_test['submission_score'])
    print(f'{round(100*(1+T_list.index(T))/len(T_list), 2)}%')

toc = time.time()
print(f'Done ({round(toc-tic, 2)} seconds)')
print('')


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Log experiment %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

current_time = time.localtime()
time_identifier = (str(current_time.tm_year) + f'{current_time.tm_mon:02d}' +
                   f'{current_time.tm_mday:02d}' +
                   f'{current_time.tm_hour:02d}' +
                   f'{current_time.tm_min:02d}' +
                   f'{current_time.tm_sec:02d}')

with open(path_to_output_data+str(time_identifier)+'_log.txt', 'a') as log:
    log.write(f'station_code = {station_code}\n')
    log.write(f'solver_IO = {solver_IO}\n')
    log.write(f'solver_complete_route = {solver_complete_route}\n')
    log.write(f'compute_train_score = {compute_train_score}\n')
    log.write(f'area_cluster = {area_cluster}\n')
    log.write(f'region_cluster = {region_cluster}\n')
    log.write(f'zone_id_diff = {zone_id_diff}\n')
    log.write(f'step_size_constant = {step_size_constant}\n')
    log.write(f'step_size_type = {step_size_type}\n')
    log.write(f'resolution = {resolution}\n')
    log.write(f'T = {T}\n')
    log.write(f'update_step = {update_step}\n')
    log.write(f'regularizer = {regularizer}\n')
    log.write(f'reg_param = {reg_param}\n')
    log.write(f'averaged_type = {averaged_type}\n')
    log.write(f'normalize_grad = {normalize_grad}\n')
    log.write(f'sub_loss = {sub_loss}\n')

results = {}

if compute_train_score:
    results['score_train_hist'] = score_train_hist
    results['len_train'] = len(dataset_train)
results['score_test_hist'] = score_test_hist
results['len_test'] = len(dataset_test)
results['T_list'] = T_list

pickle.dump(results, open(path_to_output_data + str(time_identifier)
                          + '_results.p', "wb"))

print('time_identifier = ' + str(time_identifier))
