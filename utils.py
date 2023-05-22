"""
Utils module for solving the Amazon Challenge using Inverse Optimization.

Author: Pedro Zattoni Scroccaro
"""

import os
from operator import itemgetter
from itertools import groupby
from itertools import combinations
import numpy as np
import numpy.linalg as la
from score_mod import evaluate


def zone_centers(data):
    """
    Geographical zone centers.

    Given a dataset of routes or a single route, computes geographical zone
    center of each zone in the dataset, as well as the set of stops per zone.

    Parameters
    ----------
    data : dict
        Dataset of routes, or a single route. If more than one route, the keys
        are the route IDs and the values are dictionaries with the route's
        data. If a single route, is it only the dictionary with the route's
        data, that is, without an route ID.

    Returns
    -------
    zc : dict
        Zone centers. The keys are the zone IDs (string) and the values are the
        latitude and longitude coordinates of the zone center (list)
    stops_per_zone : dict
        Set of spots of each zone in the dataset. The keys are the zone IDs
        (string) and the values are list of tuples, each tuple containing the
        latitude and longitude coordinates of a stop in that respective zone.

    """
    # Check if dataset is a single route (which is given without a route ID)
    first_key = next(iter(data))
    if first_key[:7] != 'RouteID':
        dummy_id = 'RouteID'
        data = {dummy_id: {'stops': data}}

    # Sort stops per zone
    stops_per_zone = {}
    for route_ID in data:
        stops = data[route_ID]['stops']
        for stop in stops:
            stop_data = stops[stop]
            zone_id = stop_data['zone_id']
            lat = stop_data['lat']
            lng = stop_data['lng']
            if zone_id in stops_per_zone:
                stops_per_zone[zone_id].append((lat, lng))
            else:
                stops_per_zone[zone_id] = [(lat, lng)]

    # Computes geographical center of the zones
    zc = {}
    for zone_id in stops_per_zone:
        stops = stops_per_zone[zone_id]
        center = [sum(x)/len(x) for x in zip(*stops)]
        zc[zone_id] = center

    return zc, stops_per_zone


def get_data(station_code, solver_IO, solver_complete_route,
             compute_train_score, area_cluster,
             region_cluster, zone_id_diff,
             step_size_constant, step_size_type, resolution,
             T, update_step, regularizer, reg_param,
             averaged_type, normalize_grad, sub_loss,
             path_to_output_data):
    """
    Retrieve simulation data from simulation parameters.

    Parameters
    ----------
    station_code : {'DLA7', 'DLA9', 'DLA8', 'DBO3', 'DSE5', 'DSE4', 'DCH4',
                    'DBO2', 'DCH3', 'DLA3', 'DLA4', 'DAU1', 'DCH1', 'DLA5',
                    'DSE2', 'DCH2', 'DBO1'}
        Station (depot) code.
    initial_theta : {'uniform', 'zone_centers'}
        Initial theta strategy.
    step : int
        Amazon score is computed every step iterations of the IO algorithm.
    T_max : int
        Maximum number of iterations of IO algorithm.
    step_size_constant : float
        Constant that multiplies the step-size used for the IO algorithm.
    batch : float or {'reshuffled'}
        If float, it is the ratio of dataset used in each iteration of the
        algorithm. For instance, if batch=0.1, 10% of the dataset will be used.
        If batch='reshuffled', for each iteration of the IO algorithm, it goes
        over the entire dataset one example at a time, reshuffeling the dataset
        at the beginning of each iteration.
    alpha : float
        IO paramter.
    solver_IO : {'gurobi', 'ortools', 'greedy'}
        Solver used in the IO algorithm.
    solver_complete_route : {'gurobi', 'ortools', 'greedy'}
        Solver used to compute complete route.
    route_quality : {'HML', 'HM'}
        Quality of used routes. HML means high, medium and low quality routes
        were used.
    compute_train_score : bool
        If true, also computes the score using the training data, i.e.,
        in-sample score.
    area_cluster : bool
        If true, assumes expert respects area clusters, and enforce this
        behaviour in the solution.
    region_cluster : bool
        If true, assumes expert respects region clusters, and enforce this
        behaviour in the solution..
    zone_id_diff : bool
        If true, assumes expert respects the 'one unit difference' rule, and
        enforce this behaviour in the solution.
    test_theta_init : bool
        If true, computes the score of initial theta.
    path_to_output_data : string
        Path to folder location of data.

    Returns
    -------
    time_identifier : string
        String with the time identifier of the results.

    Raises
    ------
    Exception
        If no results are found for this simulation parameters.

    """
    files = os.listdir(path_to_output_data)
    files.sort(reverse=True)

    # iterating over all files
    for file in files:
        if file.endswith('.txt'):
            time_identifier = file[:14]
            flag = True
            with open(path_to_output_data+file) as f:
                f_lines = f.readlines()
                for i in range(len(f_lines)):
                    line_splited = f_lines[i].split(' ')
                    param = line_splited[2][:-1]

                    if i == 0:
                        flag = flag*(param == station_code)
                    elif i == 1:
                        flag = flag*(param == solver_IO)
                    elif i == 2:
                        flag = flag*(param == solver_complete_route)
                    elif i == 3:
                        flag = flag*(param == str(compute_train_score))
                    elif i == 4:
                        flag = flag*(param == str(area_cluster))
                    elif i == 5:
                        flag = flag*(param == str(region_cluster))
                    elif i == 6:
                        flag = flag*(param == str(zone_id_diff))
                    elif i == 7:
                        flag = flag*(param == str(step_size_constant))
                    elif i == 8:
                        flag = flag*(param == step_size_type)
                    elif i == 9:
                        flag = flag*(param == str(resolution))
                    elif i == 10:
                        flag = flag*(param == str(T))
                    elif i == 11:
                        flag = flag*(param == update_step)
                    elif i == 12:
                        flag = flag*(param == regularizer)
                    elif i == 13:
                        flag = flag*(param == str(reg_param))
                    elif i == 14:
                        flag = flag*(param == str(averaged_type))
                    elif i == 15:
                        flag = flag*(param == str(normalize_grad))
                    elif i == 16:
                        flag = flag*(param == str(sub_loss))

                if flag:
                    return time_identifier

    raise Exception(f'No data found for {station_code}!')


def dists_test_data(theta_IO, route, zc_train, zone_id_to_index, area_cluster,
                    region_cluster, zone_id_diff):
    """
    Compute edge weights between zones for a route from the test set.

    Ideally, we would need to use only the learned IO weights to compute the
    zone sequences for the test set. However, there are two issues in the
    Amazon data. First, there are zones in the test set which are not visited
    in the training set. In these cases, for any edge connecting to an unseen
    zone, we use the euclidean distance between them as the edge weight.
    Second, there are cases when the same zone ID represents very far away
    regions in the test dataset, compared to the training dataset. In these
    case, we treat the training and test zone centers for the same zone ID as
    the same zone only if they are are close enough from each other.
    Additionally, we area cluster, region cluster, and zone ID difference
    penalizations can be used.

    Parameters
    ----------
    theta_IO : ndarray
        1D numpy array with learning IO cost vector (weights).
    route : dict
        Test route data.
    zc_train : dict
        Zone centers from training dataset.
    zone_id_to_index : dict
        Mapping from Amazon's zone IDs (string) to an unique integer.
    area_cluster : bool
        If true, assumes expert respects area clusters, and enforce this
        behaviour in the solution.
    region_cluster : bool
        If true, assumes expert respects region clusters, and enforce this
        behaviour in the solution.
    zone_id_diff : bool
        If true, assumes expert respects the 'one unit difference' rule, and
        enforce this behaviour in the solution.

    Returns
    -------
    dists : dict
        Weights to compute the zone sequence for the given test route.

    """
    m_train = len(zc_train)
    theta_IO_mat = theta_IO.reshape(m_train, m_train)

    zc_route, _ = zone_centers(route)
    new_zone_centers = set()
    for zone_id_route in zc_route:
        zc_route_lat = zc_route[zone_id_route][0]
        zc_route_lng = zc_route[zone_id_route][1]

        # Check if zone ID in test route exists in training datase, and compute
        # the distance between the zone center in the training dataset and the
        # zone center from the test route. If the zone ID does not exist in the
        # training dataset, add a dummy distance of 1 (large constant).
        try:
            zone_dist = la.norm([zc_route_lat - zc_train[zone_id_route][0],
                                 zc_route_lng - zc_train[zone_id_route][1]])
        except KeyError:
            zone_dist = 1

        # If the distance between the zone center from the traning set and the
        # zone center from the test route is larger than a predefined
        # threshold, then we treat the test zone ID as a new zone. We do this
        # because we observed inconsistensis in the geographical locations of
        # zones in the training and test datasets.
        threshold = 0.005
        if zone_dist > threshold:
            new_zone_centers.add(zone_id_route)

    # If both zones exist in the training dataset and are not considered
    # inconsistent (according to the criteria defined above), we used the
    # distances we learning with IO. Otherwise, we use their eucliden distance.
    dists = {}
    for zone1_id in zc_route:
        for zone2_id in zc_route:
            z1_new = (zone1_id in new_zone_centers)
            z2_new = (zone2_id in new_zone_centers)

            # Extract area and region from zone ID
            area1 = zone1_id[:-2]+zone1_id[-1]
            area2 = zone2_id[:-2]+zone2_id[-1]
            region1 = zone1_id[:-3]
            region2 = zone2_id[:-3]

            # Only check if zones are in the same area/region if they are both
            # new or both not new (XNOR). Otherwise, assume they are in
            # different areas/regions
            check_zone_id = z1_new*z2_new + (not z1_new)*(not z2_new)

            if area_cluster:
                if check_zone_id:
                    M_A = (area1 != area2)
                else:
                    M_A = 1
            else:
                M_A = 0

            if region_cluster:
                if check_zone_id:
                    M_R = (region1 != region2)
                else:
                    M_R = 1
            else:
                M_R = 0

            if zone_id_diff:
                W_i = ord(zone1_id[0])
                W_j = ord(zone2_id[0])
                M_W = (np.abs(W_i - W_j) > 1.5)

                try:
                    x_i = int(zone1_id[2:-3])
                    x_j = int(zone2_id[2:-3])
                    M_x = (np.abs(x_i - x_j) > 1.5)
                except ValueError:
                    M_x = 1

                try:
                    y_i = int(zone1_id[-2])
                    y_j = int(zone2_id[-2])
                    M_y = (np.abs(y_i - y_j) > 1.5)
                except ValueError:
                    M_y = 1
                    if (zone1_id != 'depot') and (zone2_id != 'depot'):
                        print(zone1_id, zone2_id)

                Z_i = ord(zone1_id[-1])
                Z_j = ord(zone2_id[-1])
                M_Z = (np.abs(Z_i - Z_j) > 1.5)

                M_diff = M_W + M_x + M_y + M_Z
            else:
                M_diff = 0

            M_pen = M_A + M_R + M_diff

            if z1_new and z2_new:
                dists[(zone1_id, zone2_id)] = M_pen \
                    + la.norm([zc_route[zone1_id][0] - zc_route[zone2_id][0],
                               zc_route[zone1_id][1] - zc_route[zone2_id][1]])
            elif z1_new and (not z2_new):
                dists[(zone1_id, zone2_id)] = M_pen \
                    + la.norm([zc_route[zone1_id][0] - zc_train[zone2_id][0],
                               zc_route[zone1_id][1] - zc_train[zone2_id][1]])
            elif (not z1_new) and z2_new:
                dists[(zone1_id, zone2_id)] = M_pen \
                    + la.norm([zc_train[zone1_id][0] - zc_route[zone2_id][0],
                               zc_train[zone1_id][1] - zc_route[zone2_id][1]])
            elif (not z1_new) and (not z2_new):
                dists[(zone1_id, zone2_id)] = M_pen \
                    + theta_IO_mat[zone_id_to_index[zone1_id]
                                   ][zone_id_to_index[zone2_id]]

    return dists


def amazon_score(theta_IO, dataset, zc_train, zone_id_to_index,
                 index_to_zone_id, solver_complete_route, solver_IO,
                 station_code, area_cluster, region_cluster, zone_id_diff):
    """
    Compute amazon score.

    Computes zone sequence using IO learned cost vector. From this zone
    sequence, computes complete route at the stop (customer) level. Finally,
    computes the Amazon score of the final complete route by comparing it with
    the true route.

    Parameters
    ----------
    theta_IO : ndarray
        1D numpy array with learning IO cost vector (weights).
    dataset : dict
        Dataset of route examples.
    zc_train : dict
        Zone centers from training dataset.
    zone_id_to_index : dict
        Mapping from Amazon's zone IDs (string) to an unique integer.
    solver_complete_route : {'gurobi', 'ortools', 'greedy'}
        Solver used to compute complete route.
    solver_IO : {'gurobi', 'ortools', 'greedy'}
        Solver used in the IO algorithm.
    station_code : {'DLA7', 'DLA9', 'DLA8', 'DBO3', 'DSE5', 'DSE4', 'DCH4',
                    'DBO2', 'DCH3', 'DLA3', 'DLA4', 'DAU1', 'DCH1', 'DLA5',
                    'DSE2', 'DCH2', 'DBO1'}
        Station (depot) code.
    path_to_output_data : string
        Path to folder location of data.
    area_cluster : bool
        If true, assumes expert respects area clusters, and enforce this
        behaviour in the solution.
    region_cluster : bool
        If true, assumes expert respects region clusters, and enforce this
        behaviour in the solution.
    zone_id_diff : bool
        If true, assumes expert respects the 'one unit difference' rule, and
        enforce this behaviour in the solution.

    Returns
    -------
    scores : dict
        Amazon score for the given dataset, as well as the individual score of
        each route.

    """
    scores = {}
    submission = {}
    actual_routes = {}
    cost_matrices = {}
    invalid_scores = {}
    for route_ID in dataset:
        route = dataset[route_ID]['stops'].copy()
        route_tt = dataset[route_ID]['travel_times'].copy()
        stop_seq_true = route_to_stop_seq(route)

        dists_mod = dists_test_data(theta_IO, route, zc_train,
                                    zone_id_to_index, area_cluster,
                                    region_cluster, zone_id_diff)

        # Compute zone sequence
        zone_id_seq_mod, _ = solve_ATSP(dists_mod, solver_IO)
        zone_id_seq = [zone_id if zone_id[0] != zone_id[1] else zone_id[1:]
                       for zone_id in zone_id_seq_mod]

        # Compute complete stop sequence
        stop_seq = zone_seq_to_stop_seq(zone_id_seq, route, route_tt,
                                        solver_complete_route)

        # Put data in the required format
        indexes = list(range(len(stop_seq)))
        stop_seq_dict = dict(zip(stop_seq, indexes))
        submission[route_ID] = {'proposed': stop_seq_dict.copy()}

        stop_seq_true_dict = dict(zip(stop_seq_true, indexes))
        actual_routes[route_ID] = {'actual': stop_seq_true_dict.copy()}

        cost_matrices[route_ID] = route_tt
        invalid_scores[route_ID] = dataset[route_ID]['invalid_sequence_score']

    # Compute Amazon score
    scores = evaluate(actual_routes, submission, cost_matrices, invalid_scores)

    return scores


def zone_seq_to_stop_seq(zone_id_seq, route, route_tt, solver_complete_route):
    """
    Compute complete stop sequence from learned zone sequence.

    Parameters
    ----------
    zone_id_seq : list
        List of zone IDs in the order they should be visited.
    route : dict
        Route data.
    route_tt : dict
        Travel times between each stop of the route.
    solver_complete_route : {'gurobi', 'ortools', 'greedy'}
        Solver used to compute complete route.

    Returns
    -------
    stop_seq_rot : list
        Resulting sequence of stops, starting from the depot.

    """
    # Panlization constant used to enforce the zone sequence
    M = 1e4

    # Compute penized distances
    dists = {}
    for stop1 in route_tt:
        for stop2 in route_tt:
            zone1_id = route[stop1]['zone_id']
            zone2_id = route[stop2]['zone_id']
            if zone1_id == 'depot':
                depot_stop = stop1
            if stop1 != stop2:
                if zone1_id == zone2_id:
                    dists[(stop1, stop2)] = route_tt[stop1][stop2]
                else:
                    zone1_seq_idx = zone_id_seq.index(zone1_id)
                    zone2_seq_idx = zone_id_seq.index(zone2_id)
                    if ((zone2_seq_idx == zone1_seq_idx+1)
                        or (zone2_seq_idx == 0
                            and zone1_seq_idx == len(zone_id_seq)-1)):

                        dists[(stop1, stop2)] = route_tt[stop1][stop2] + M
                    else:
                        dists[(stop1, stop2)] = route_tt[stop1][stop2] + 2*M

    # Compute complete route
    stop_seq, _ = solve_ATSP(dists, solver_complete_route)

    # Rotate list so that the depot is the first stop
    depot_idx = stop_seq.index(depot_stop)
    stop_seq_rot = stop_seq[depot_idx:] + stop_seq[:depot_idx]

    return stop_seq_rot


def zone_seq_to_vec(zone_seq, m):
    """
    Transform a zone sequence from a list of indexes to a vector.

    Given a list of zones encoded by their unique integer index, creates a
    matrix, where the element (i,j) of the matrix equals to 1 if zone
    j is visited after zone i, and 0 otherwise. Return the flattened version
    of this matrix.

    Parameters
    ----------
    zone_seq : list
        Zone sequence. The zones are encoded using their
        unique integer index, intead of thier Amazon zone ID.
    m : int
        Total number of zones. Used to define the size of the matrix.

    Returns
    -------
    x_vec : ndarray
        1D numpy array corresponding to the flattened matrix binary matrix
        encoding the zone sequence.

    """
    n = len(zone_seq)

    x_mat = np.zeros((m, m))
    x_mat[zone_seq[n-1], zone_seq[0]] = 1
    for i in range(n-1):
        x_mat[zone_seq[i], zone_seq[i+1]] = 1

    x_vec = x_mat.flatten()
    return x_vec


def route_to_stop_seq(route):
    """
    Extract stop sequence from route data.

    Parameters
    ----------
    route : dict
        Route data.

    Returns
    -------
    stop_seq_sorted : list
        List of stops sorted by their sequence index.

    """
    # Extract sequence index
    stop_seq_idx = []
    for stop in route:
        sequence_index = route[stop]['sequence_index']
        stop_seq_idx.append((stop, sequence_index))

    # Sort by sequence index
    stop_seq_sorted = sorted(stop_seq_idx, key=itemgetter(1))
    stop_seq_sorted = [item[0] for item in stop_seq_sorted]

    return stop_seq_sorted


def route_to_zone_seq(route, zone_id_to_index=None):
    """
    Extract zone sequence from route data.

    Transform the sequence of stop into a sequence of zone ID, using the zone
    of the corresponding stop in the route data. Equal zone IDs are merged if
    cosecutive. If the same zone ID appeard multiple times after merging, keep
    the one with the most consecutive stops.

    Parameters
    ----------
    route : dict
        Route data.
    zone_id_to_index : dict, optional
        Mapping from Amazon's zone IDs (string) to a unique integer index.

    Returns
    -------
    zone_seq : list
        Resulting zone sequence. If zone_id_to_index is given, zones are
        encoded using unique integer index. If zone_id_to_index is None, uses
        Amazon's original zone ID string.

    """
    # Extract sequence index of zones
    zone_sequence_index = []
    for stop in route:
        zone_id = route[stop]['zone_id']
        sequence_index = route[stop]['sequence_index']
        zone_sequence_index.append((zone_id, sequence_index))

    # Sort zone sequence
    zone_sequence_sorted = sorted(zone_sequence_index, key=itemgetter(1))
    zone_sequence_sorted = [item[0] for item in zone_sequence_sorted]

    # Merge and count consecutive apperances
    zone_sequence_sorted_merged = [(key, sum(1 for i in g)) for key, g
                                   in groupby(zone_sequence_sorted)]

    # Find the size of longest sequence of the same zone, for each zone ID
    zone_id_set = set([key for key, _group in groupby(zone_sequence_sorted)])
    zone_longest = {zone: 0 for zone in zone_id_set}
    for zone, lengh in zone_sequence_sorted_merged:
        longest = zone_longest[zone]
        if lengh > longest:
            zone_longest[zone] = lengh

    # Keep only the zone ID of the longest sequence
    zone_seq = []
    for zone, lengh in zone_sequence_sorted_merged:
        if lengh == zone_longest[zone]:
            zone_seq.append(zone)

    # Map zone IDs to zone indexes
    if zone_id_to_index is not None:
        zone_seq = [zone_id_to_index[zone] for zone in zone_seq]

    return zone_seq


def ATSP_gurobi(dists):
    """
    Asymmetric TSP solver using Gurobi.

    Adapted from https://www.gurobi.com/documentation/9.5/examples/tsp_py.html.
    See also: https://gurobi.github.io/modeling-examples/traveling_salesman/tsp.html
    or https://colab.research.google.com/github/Gurobi/modeling-examples/blob/master/traveling_salesman/tsp_gcl.ipynb
    """
    import gurobipy as gp

    # Callback - use lazy constraints to eliminate sub-tours
    def subtourelim(model, where):
        if where == gp.GRB.Callback.MIPSOL:
            vals = model.cbGetSolution(model._vars)
            selected = gp.tuplelist((i, j)
                                    for i, j in model._vars.keys()
                                    if vals[i, j] > 0.5)
            # find the shortest cycle in the selected edge list
            tour = subtour(selected)
            if len(tour) < n:
                # add subtour elimination constraint for every pair of cities
                # in tour
                model.cbLazy(gp.quicksum(model._vars[i, j] + model._vars[j, i]
                                         for i, j in combinations(tour, 2))
                             <= len(tour)-1)

    def subtour(edges):
        unvisited = list(nodes)
        cycle = list(nodes)  # Dummy - guaranteed to be replaced
        while unvisited:  # true if list is non-empty
            thiscycle = []
            neighbors = unvisited
            while neighbors:
                current = neighbors[0]
                thiscycle.append(current)
                unvisited.remove(current)
                neighbors = [j for i, j in edges.select(current, '*')
                             if j in unvisited]
            if len(thiscycle) <= len(cycle):
                cycle = thiscycle  # New shortest subtour
        return cycle

    # When using Gurobi solver, dists dictionary should not contain the
    # distance from a node to itself.
    dists = {(i, j): dists[(i, j)] for i, j in dists if i != j}

    nodes = set([node for node, _ in dists.keys()])
    n = len(nodes)
    mdl = gp.Model()
    mdl.setParam('OutputFlag', 0)

    # Create variables
    vs = mdl.addVars(dists.keys(), obj=dists, vtype=gp.GRB.BINARY, name='e')

    mdl.addConstrs(vs.sum(node, '*') == 1 for node in nodes)
    mdl.addConstrs(vs.sum('*', node) == 1 for node in nodes)

    # Optimize model
    mdl._vars = vs
    mdl.Params.LazyConstraints = 1
    mdl.Params.TimeLimit = 600

    mdl.optimize(subtourelim)

    try:
        vals = mdl.getAttr('x', vs)
        selected = gp.tuplelist((i, j)
                                for i, j in vals.keys() if vals[i, j] > 0.5)
        tour = subtour(selected)
        obj_val = mdl.ObjVal
    except Exception:
        print(f'Not able to solve ATSP in {mdl.Params.TimeLimit} seconds ' +
              'with gurobi!')
        print('Solving it with ortools.')
        tour, obj_val = ATSP_ortools(dists)

    if len(tour) != n:
        print('Gurobi: tour length different than number of nodes required ' +
              'to be visited!')
        print('Solving it with ortools.')
        tour, obj_val = ATSP_ortools(dists)

    return tour, obj_val


def ATSP_ortools(dists):
    """
    Asymmetric TSP solver using Google OR-Tools.

    Adapted from https://developers.google.com/optimization/routing/tsp
    """
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp

    def get_solution(manager, routing, solution):
        index = routing.Start(0)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))
        return route

    def distance_callback(from_index, to_index):
        """Return the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dists_new[from_node][to_node]

    # OR-Tools does not accept arbitrary node labels. We need to remap them
    # so if the problem as n nodes, labeled from 1 to n.
    list_nodes = list(set([node for node, _ in dists]))
    n = len(list_nodes)
    indexes = list(range(n))
    idx_to_zone = dict(zip(indexes, list_nodes))

    scale_factor = 1e4
    dists_new = {}
    for i in indexes:
        dists_new[i] = {}
        for j in indexes:
            if i == j:
                dists_new[i][j] = 0
            else:
                dists_new[i][j] = int(
                    scale_factor*dists[idx_to_zone[i], idx_to_zone[j]])

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    try:
        route = get_solution(manager, routing, solution)
        obj_val = solution.ObjectiveValue()/scale_factor
        # get original nodes back
        route = [idx_to_zone[route[i]] for i in range(len(route)-1)]
    except Exception:
        print('ortools failed. Returning greedy tour.')
        route, obj_val = ATSP_greedy(dists)

    return route, obj_val


def ATSP_greedy(dists_dict):
    """
    Asymmetric TSP greedy solver.

    Adapted from https://www.dcc.fc.up.pt/~jpp/code/py_metaheur/tsp.py
    """
    def mk_closest(dists, n):
        """Compute a sorted list of the distances for each of the nodes.

        For each node, the entry is in the form [(d1,i1), (d2,i2), ...]
        where each tuple is a pair (distance,node).
        """
        C = []
        for i in range(n):
            dlist = [(dists[i][j], j) for j in range(n) if j != i]
            dlist.sort()
            C.append(dlist)
        return C

    def length(tour, dists):
        """Calculate the length of a tour according to distance matrix 'D'."""
        z = dists[tour[-1]][tour[0]
                            ]    # edge from last to first city of the tour
        for i in range(1, len(tour)):
            # add length of edge from city i-1 to i
            z += dists[tour[i]][tour[i-1]]
        return z

    def nearest(last, unvisited, dists):
        """Return the index of the node which is closest to 'last'."""
        near = unvisited[0]
        min_dist = dists[last][near]
        for i in unvisited[1:]:
            if dists[last][i] < min_dist:
                near = i
                min_dist = dists[last][near]
        return near

    def nearest_neighbor(n, i, dists):
        """Return tour starting from city 'i', using the Nearest Neighbor.

        Uses the Nearest Neighbor heuristic to construct a solution:
        - start visiting city i
        - while there are unvisited cities, follow to the closest one
        - return to city i
        """
        unvisited = list(range(n))
        unvisited.remove(i)
        last = i
        tour = [i]
        while unvisited != []:
            next = nearest(last, unvisited, dists)
            tour.append(next)
            unvisited.remove(next)
            last = next
        return tour

    # just like for the OR tools solver, we remap them
    # so if the problem as n nodes, the labels are from 1 to n.
    list_nodes = list(set([node for node, _ in dists_dict]))
    n = len(list_nodes)
    indexes = list(range(n))
    idx_to_zone = dict(zip(indexes, list_nodes))

    scale_factor = 1e6
    dists = [[] for _ in indexes]
    for i in indexes:
        for j in indexes:
            if i == j:
                dists[i].append(0)
            else:
                dists[i].append(round(scale_factor*dists_dict[idx_to_zone[i],
                                                              idx_to_zone[j]]))

    # create a greedy tour, visiting city '0' first
    tour = nearest_neighbor(n, 0, dists)
    obj_val = length(tour, dists)/scale_factor

    # get original nodes back
    tour = [idx_to_zone[tour[i]] for i in range(len(tour)-1)]

    return tour, obj_val


def solve_ATSP(dists, solver):
    """Wrap available ATSP solvers."""
    if solver == 'gurobi':
        route, obj_val = ATSP_gurobi(dists)
    elif solver == 'ortools':
        route, obj_val = ATSP_ortools(dists)
    elif solver == 'greedy':
        route, obj_val = ATSP_greedy(dists)

    return route, obj_val
