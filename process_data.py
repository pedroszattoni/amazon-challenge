"""
Preprocess and split Amazon Challange data, creating one dataset per depot.

Author: Pedro Zattoni Scroccaro
"""

import json
import pickle


def closest(stop, stops):
    """
    Clostes stop.

    Compute euclidean distance from all other stops in the route and asign
    zone ID of closes zone with valid zone ID
    """
    lat = stops[stop]['lat']
    lng = stops[stop]['lng']
    dists = {}
    for stop1 in stops:
        zone_id1 = stops[stop1]['zone_id']
        if (stop1 != stop) and isinstance(zone_id1, str):
            lat1 = stops[stop1]['lat']
            lng1 = stops[stop1]['lng']
            dists[stop1] = (lat - lat1)**2 + (lng - lng1)**2
    closest_stop = min(dists, key=dists.get)
    return closest_stop


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Load data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

path_to_input_data = 'path/to/input/data/'

# Import all data from the data files
f = open(path_to_input_data + 'package_data.json')
package_data = json.load(f)

f = open(path_to_input_data + 'actual_sequences.json')
actual_sequences_train = json.load(f)
f = open(path_to_input_data + 'invalid_sequence_scores.json')
invalid_sequence_scores_train = json.load(f)
f = open(path_to_input_data + 'route_data.json')
route_data_train = json.load(f)
f = open(path_to_input_data + 'travel_times.json')
travel_times_train = json.load(f)

f = open(path_to_input_data + 'eval_actual_sequences.json')
actual_sequences_test = json.load(f)
f = open(path_to_input_data + 'eval_invalid_sequence_scores.json')
invalid_sequence_scores_test = json.load(f)
f = open(path_to_input_data + 'eval_route_data.json')
route_data_test = json.load(f)
f = open(path_to_input_data + 'eval_travel_times.json')
travel_times_test = json.load(f)


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Process data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# station_codes = ['DAU1', 'DBO1', 'DBO2', 'DBO3', 'DCH1', 'DCH2', 'DCH3',
#                  'DCH4', 'DLA3', 'DLA4', 'DLA5', 'DLA7', 'DLA8', 'DLA9',
#                  'DSE2', 'DSE4', 'DSE5']

DAU1_train = {}
DBO1_train = {}
DBO2_train = {}
DBO3_train = {}
DCH1_train = {}
DCH2_train = {}
DCH3_train = {}
DCH4_train = {}
DLA3_train = {}
DLA4_train = {}
DLA5_train = {}
DLA7_train = {}
DLA8_train = {}
DLA9_train = {}
DSE2_train = {}
DSE4_train = {}
DSE5_train = {}
DAU1_test = {}
DBO1_test = {}
DBO2_test = {}
DBO3_test = {}
DCH1_test = {}
DCH2_test = {}
DCH3_test = {}
DCH4_test = {}
DLA3_test = {}
DLA4_test = {}
DLA5_test = {}
DLA7_test = {}
DLA8_test = {}
DLA9_test = {}
DSE2_test = {}
DSE4_test = {}
DSE5_test = {}

for route_ID in route_data_train:
    stops = route_data_train[route_ID]['stops']
    station_code = route_data_train[route_ID]['station_code']

    # Add sequence index to route data, add change depot's zone ID to 'depot'
    # and give closes stop zone ID to stops without zone ID
    for stop in stops:
        stops[stop]['sequence_index'] = (actual_sequences_train[route_ID]
                                         ['actual'][stop])

        zone_id = stops[stop]['zone_id']
        depot = (stops[stop]['type'] == 'Station')

        # assign zones to stops with no zone ID. Also, depot dummy zone
        if (not isinstance(zone_id, str)) and (not depot):
            closest_stop = closest(stop, stops)
            stops[stop]['zone_id'] = stops[closest_stop]['zone_id']
        elif depot:
            stops[stop]['zone_id'] = 'depot'

    # Check if route contains packages that needed to be redelivered
    delivered = True
    for stop in package_data[route_ID]:
        for package in package_data[route_ID][stop]:
            if (package_data[route_ID][stop][package]['scan_status']
                    == 'DELIVERY_ATTEMPTED'):
                delivered = False
                break
        if not delivered:
            break

    route_data = {}
    route_data['stops'] = stops
    route_data['travel_times'] = travel_times_train[route_ID]
    route_data['invalid_sequence_score'] \
        = invalid_sequence_scores_train[route_ID]
    route_data['route_score'] = route_data_train[route_ID]['route_score']
    route_data['delivered'] = delivered

    if station_code == 'DAU1':
        DAU1_train[route_ID] = route_data
    elif station_code == 'DBO1':
        DBO1_train[route_ID] = route_data
    elif station_code == 'DBO2':
        DBO2_train[route_ID] = route_data
    elif station_code == 'DBO3':
        DBO3_train[route_ID] = route_data
    elif station_code == 'DCH1':
        DCH1_train[route_ID] = route_data
    elif station_code == 'DCH2':
        DCH2_train[route_ID] = route_data
    elif station_code == 'DCH3':
        DCH3_train[route_ID] = route_data
    elif station_code == 'DCH4':
        DCH4_train[route_ID] = route_data
    elif station_code == 'DLA3':
        DLA3_train[route_ID] = route_data
    elif station_code == 'DLA4':
        DLA4_train[route_ID] = route_data
    elif station_code == 'DLA5':
        DLA5_train[route_ID] = route_data
    elif station_code == 'DLA7':
        DLA7_train[route_ID] = route_data
    elif station_code == 'DLA8':
        DLA8_train[route_ID] = route_data
    elif station_code == 'DLA9':
        DLA9_train[route_ID] = route_data
    elif station_code == 'DSE2':
        DSE2_train[route_ID] = route_data
    elif station_code == 'DSE4':
        DSE4_train[route_ID] = route_data
    elif station_code == 'DSE5':
        DSE5_train[route_ID] = route_data

for route_ID in route_data_test:
    station_code = route_data_test[route_ID]['station_code']
    stops = route_data_test[route_ID]['stops']

    for stop in stops:
        stops[stop]['sequence_index'] = (actual_sequences_test[route_ID]
                                         ['actual'][stop])

        zone_id = stops[stop]['zone_id']
        depot = (stops[stop]['type'] == 'Station')

        if (not isinstance(zone_id, str)) and (not depot):
            closest_stop = closest(stop, stops)
            stops[stop]['zone_id'] = stops[closest_stop]['zone_id']
        elif depot:
            stops[stop]['zone_id'] = 'depot'

    route_data = {}
    route_data['stops'] = stops
    route_data['travel_times'] = travel_times_test[route_ID]
    route_data['invalid_sequence_score'] \
        = invalid_sequence_scores_test[route_ID]

    if station_code == 'DAU1':
        DAU1_test[route_ID] = route_data
    elif station_code == 'DBO1':
        DBO1_test[route_ID] = route_data
    elif station_code == 'DBO2':
        DBO2_test[route_ID] = route_data
    elif station_code == 'DBO3':
        DBO3_test[route_ID] = route_data
    elif station_code == 'DCH1':
        DCH1_test[route_ID] = route_data
    elif station_code == 'DCH2':
        DCH2_test[route_ID] = route_data
    elif station_code == 'DCH3':
        DCH3_test[route_ID] = route_data
    elif station_code == 'DCH4':
        DCH4_test[route_ID] = route_data
    elif station_code == 'DLA3':
        DLA3_test[route_ID] = route_data
    elif station_code == 'DLA4':
        DLA4_test[route_ID] = route_data
    elif station_code == 'DLA5':
        DLA5_test[route_ID] = route_data
    elif station_code == 'DLA7':
        DLA7_test[route_ID] = route_data
    elif station_code == 'DLA8':
        DLA8_test[route_ID] = route_data
    elif station_code == 'DLA9':
        DLA9_test[route_ID] = route_data
    elif station_code == 'DSE2':
        DSE2_test[route_ID] = route_data
    elif station_code == 'DSE4':
        DSE4_test[route_ID] = route_data
    elif station_code == 'DSE5':
        DSE5_test[route_ID] = route_data

# %%%%%%%%%%%%%%%%%%%%%%% Save processed data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pickle.dump(DAU1_train, open(path_to_input_data + 'DAU1_train.p', "wb"))
pickle.dump(DBO1_train, open(path_to_input_data + 'DBO1_train.p', "wb"))
pickle.dump(DBO2_train, open(path_to_input_data + 'DBO2_train.p', "wb"))
pickle.dump(DBO3_train, open(path_to_input_data + 'DBO3_train.p', "wb"))
pickle.dump(DCH1_train, open(path_to_input_data + 'DCH1_train.p', "wb"))
pickle.dump(DCH2_train, open(path_to_input_data + 'DCH2_train.p', "wb"))
pickle.dump(DCH3_train, open(path_to_input_data + 'DCH3_train.p', "wb"))
pickle.dump(DCH4_train, open(path_to_input_data + 'DCH4_train.p', "wb"))
pickle.dump(DLA3_train, open(path_to_input_data + 'DLA3_train.p', "wb"))
pickle.dump(DLA4_train, open(path_to_input_data + 'DLA4_train.p', "wb"))
pickle.dump(DLA5_train, open(path_to_input_data + 'DLA5_train.p', "wb"))
pickle.dump(DLA7_train, open(path_to_input_data + 'DLA7_train.p', "wb"))
pickle.dump(DLA8_train, open(path_to_input_data + 'DLA8_train.p', "wb"))
pickle.dump(DLA9_train, open(path_to_input_data + 'DLA9_train.p', "wb"))
pickle.dump(DSE2_train, open(path_to_input_data + 'DSE2_train.p', "wb"))
pickle.dump(DSE4_train, open(path_to_input_data + 'DSE4_train.p', "wb"))
pickle.dump(DSE5_train, open(path_to_input_data + 'DSE5_train.p', "wb"))

pickle.dump(DAU1_test, open(path_to_input_data + 'DAU1_test.p', "wb"))
pickle.dump(DBO1_test, open(path_to_input_data + 'DBO1_test.p', "wb"))
pickle.dump(DBO2_test, open(path_to_input_data + 'DBO2_test.p', "wb"))
pickle.dump(DBO3_test, open(path_to_input_data + 'DBO3_test.p', "wb"))
pickle.dump(DCH1_test, open(path_to_input_data + 'DCH1_test.p', "wb"))
pickle.dump(DCH2_test, open(path_to_input_data + 'DCH2_test.p', "wb"))
pickle.dump(DCH3_test, open(path_to_input_data + 'DCH3_test.p', "wb"))
pickle.dump(DCH4_test, open(path_to_input_data + 'DCH4_test.p', "wb"))
pickle.dump(DLA3_test, open(path_to_input_data + 'DLA3_test.p', "wb"))
pickle.dump(DLA4_test, open(path_to_input_data + 'DLA4_test.p', "wb"))
pickle.dump(DLA5_test, open(path_to_input_data + 'DLA5_test.p', "wb"))
pickle.dump(DLA7_test, open(path_to_input_data + 'DLA7_test.p', "wb"))
pickle.dump(DLA8_test, open(path_to_input_data + 'DLA8_test.p', "wb"))
pickle.dump(DLA9_test, open(path_to_input_data + 'DLA9_test.p', "wb"))
pickle.dump(DSE2_test, open(path_to_input_data + 'DSE2_test.p', "wb"))
pickle.dump(DSE4_test, open(path_to_input_data + 'DSE4_test.p', "wb"))
pickle.dump(DSE5_test, open(path_to_input_data + 'DSE5_test.p', "wb"))
