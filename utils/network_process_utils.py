import pypsa
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
import networkx as nx
import pandas as pd
from pypower.api import ext2int, ppoption, runopf, rundcopf, runpf
from pypower import idx_bus, idx_gen, idx_brch, idx_dcline
from pypower.api import makeYbus
import pytz
from pytz import country_timezones

def calculate_intersections(x1, y1, x2, y2, resolution=0.25):
    offset = resolution / 2
    # Initialize the list of intersection points with the endpoints included
    points = [(x1, y1), (x2, y2)]

    # Calculate the range of grid lines to check
    min_x = min(x1, x2)
    max_x = max(x1, x2)
    min_y = min(y1, y2)
    max_y = max(y1, y2)

    # Calculate intersections with vertical grid lines
    x_start = np.floor((min_x - offset) / resolution) * resolution + offset
    x_end = np.ceil((max_x - offset) / resolution) * resolution + offset
    for x in np.arange(x_start, x_end + resolution, resolution):
        if x1 != x2:  # Avoid division by zero
            y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
            if min_y <= y <= max_y and min_x <= x <= max_x:
                points.append((x, y))

    # Calculate intersections with horizontal grid lines
    y_start = np.floor((min_y - offset) / resolution) * resolution + offset
    y_end = np.ceil((max_y - offset) / resolution) * resolution + offset
    for y in np.arange(y_start, y_end + resolution, resolution):
        if y1 != y2:  # Avoid division by zero
            x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if min_y <= y <= max_y and min_x <= x <= max_x:
                points.append((x, y))

    # Remove duplicates and sort the points
    points = list(set(points))
    points.sort(key=lambda p: np.hypot(p[0]-x1, p[1]-y1))

    return points

def visualize_intersections(points, segment_info, resolution=0.25):
    x_vals, y_vals = zip(*points)
    plt.figure(figsize=(10, 10))
    plt.plot(x_vals, y_vals, 'b-', label='Line Segment')
    plt.scatter(x_vals, y_vals, color='red', s=100, zorder=5, label='Intersection Points')

    # Calculate grid boundaries for visualization
    grid_x_start = np.floor(min(x_vals) / resolution) * resolution + 0.125
    grid_x_end = np.ceil(max(x_vals) / resolution) * resolution + 0.125
    grid_y_start = np.floor(min(y_vals) / resolution) * resolution + 0.125
    grid_y_end = np.ceil(max(y_vals) / resolution) * resolution + 0.125

    # Draw grid lines and plot centers
    for x in np.arange(grid_x_start, grid_x_end, resolution):
        plt.axvline(x, color='gray', linestyle='--', linewidth=0.5)
    for y in np.arange(grid_y_start, grid_y_end, resolution):
        plt.axhline(y, color='gray', linestyle='--', linewidth=0.5)
    # for x in np.arange(grid_x_start-0.125, grid_x_end+0.125, resolution):
    #     for y in np.arange(grid_y_start-0.125, grid_y_end+0.125, resolution):
            # plt.scatter(x, y, color='blue', s=10, zorder=10)  # Plot grid centers
    segment_info = np.array(segment_info)
    plt.scatter(segment_info[:,0], segment_info[:,1], s=15, zorder=10)

    plt.grid(False)
    plt.title('Intersection of Line with Grid')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.show()

def calculate_segment_info(points, spacing):
    [x1,y1], [x2,y2] = points[0], points[-1]
    total_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    segment_info = []

    for i in range(len(points) - 1):
        px1, py1 = points[i]
        px2, py2 = points[i+1]
        # pxm = (px1+px2)/2
        # pym = (py1+py2)/2
        # xc = int(np.floor(pxm / spacing))
        # yc = int(np.floor(pym / spacing))

        segment_length = np.sqrt((px2 - px1)**2 + (py2 - py1)**2)
        proportion_length = segment_length / total_length
        nearest_center_x = (np.round((px1 + px2) / (2 * spacing)) * spacing) 
        nearest_center_y = (np.round((py1 + py2) / (2 * spacing)) * spacing) 
        segment_info.append([nearest_center_x, nearest_center_y, proportion_length])

    return segment_info

def divide_segments(network, seg_len=25, grid_space=0.25):
    max_num_seg = int(max(network.lines['length']) / seg_len + 1) * 2
    network.segments = np.zeros([network.lines.shape[0], max_num_seg, 3])
    for i, line_id in enumerate(network.lines.index):
        line_item = network.lines.loc[line_id]
        fx, fy = network.buses.loc[line_item['bus0'], ['x', 'y']]
        tx, ty = network.buses.loc[line_item['bus1'], ['x', 'y']]
        intersections = calculate_intersections(fx, fy, tx, ty, grid_space) 
        segment_info = calculate_segment_info(intersections, grid_space) # [x,y,len_ratio]
        network.lines.loc[line_id, 'num_seg'] = len(segment_info)
        network.segments[i, :len(segment_info)] = segment_info
        # visualize_intersections(intersections, segment_info, resolution=0.25)
    return network

def get_timezone_difference(country_code, utc_datetime):
    """
    Returns the timezone offset from UTC for a given country code and UTC datetime.

    Args:
    country_code (str): Two-letter ISO 3166-1 country code.
    utc_datetime (datetime): UTC datetime object.

    Returns:
    int: Timezone offset from UTC in hours. Positive for east of UTC, negative for west.
    """

    # Check if the country code is valid
    if country_code not in country_timezones:
        raise ValueError("Invalid country code")

    # Ensure the datetime is in UTC
    if utc_datetime.tzinfo is None or utc_datetime.tzinfo.utcoffset(utc_datetime) is None:
        utc_datetime = pytz.utc.localize(utc_datetime)

    # Get timezones for the country
    timezones = country_timezones[country_code]

    # If there are multiple timezones, select the first one (common case)
    tz = pytz.timezone(timezones[0])
    local_dt = utc_datetime.astimezone(tz)

    # Calculate offset in hours from UTC
    offset = local_dt.utcoffset().total_seconds() / 3600

    return int(offset)

def pypsa_pypower(network, args = None):
    phase_factor = args['phase_factor']
    date = args['date']
    baseMVA = args['BaseMVA']
    """
    Country code
    """
    bus = network.buses
    """
    Generator data
    bus	Pg	Qg	Qmax Qmin Vg	mBase (Pnom) status Pmax Pmin Extendable
    Generator cost ata
    2	startup	shutdown n	c(n-1)	...	c0
    """
    Gen_data = []
    Gen_cost_data = []
    Renewable_data = []
    generators = network.generators
    p_max_pu = network.generators_t['p_max_pu']
    renewable_gen_list = p_max_pu.columns
    p_min_pu = network.generators_t['p_min_pu']
    gen_id = generators.index.to_list()
    PV_bus = []
    for _, gen_name in enumerate(gen_id):
        bus_name = generators.loc[gen_name, 'bus']
        PV_bus.append(bus_name)
        bus_index = network.buses.index.tolist().index(bus_name)
        efficiency = generators.loc[gen_name, 'efficiency']
        p_nom = generators.loc[gen_name, 'p_nom'] / phase_factor
        # print(p_nom, generators.loc[gen_name, 'p_max_pu'])
        Pmax = generators.loc[gen_name, 'p_max_pu'] * p_nom
        Pmin = generators.loc[gen_name, 'p_min_pu'] * p_nom
        if gen_name in renewable_gen_list:
            if not args['renewable_mode']:
                Pmax = Pmin = 0
            # elif args['renewable_mode'] == 1:
            #     Pmax = Pmin = p_max_pu[gen_name][date] * p_nom
            # elif args['renewable_mode'] == 2:
            #     Pmax = p_max_pu[gen_name][date] * p_nom
            #     Pmin = 0
        if args['extendable_gen_capacity']:
            Extendable = generators.loc[gen_name, 'p_nom_extendable']
        else:
            Extendable = False
        Qmax = Pmax * args['reactive_gen_upper'] 
        Qmin = Pmax * args['reactive_gen_lower'] 
        Vg = 1
        status = 1#generators.loc[gen_name, 'status']
        Pg = 0#(Pmax+Pmin)/2
        Qg = 0#(Qmax+Qmin)/2
        Gen_data.append([bus_index, Pg, Qg, Qmax, Qmin, Vg, p_nom, status, Pmax, Pmin, Extendable])
        marginal_cost_quadratic = 0 #generators.loc[gen_name, 'marginal_cost_quadratic']
        marginal_cost = generators.loc[gen_name, 'marginal_cost']
        constant_cost = 0
        extend_cost = generators.loc[gen_name, 'capital_cost']
        Gen_cost_data.append([2, 0, 0, 3, marginal_cost_quadratic, marginal_cost, constant_cost, extend_cost])

    """
    Bus data
    bus_i	type Pd	Qd	Gs	Bs	area Vm	Va	baseKV zone Vmax Vmin busx busy
    """
    bus = network.buses
    shunt_elem = network.shunt_impedances
    Bus_data = []
    pop_ratio = network.buses['pop_ratio']
    slack_bus = network.sub_networks['slack_bus']
    for i, bus_name in enumerate(network.buses.index):
        # control_type = bus.loc[bus_name, 'control']
        if bus_name in slack_bus:
            bus_type = 3
        else:
            if bus_name in PV_bus: # PV bus
                bus_type = 2
            else: # PQ bus
                bus_type = 1
        bus_index = i
        Pd = network.buses.loc[bus_name, 'p_set'] / args['phase_factor']
        Qd = network.buses.loc[bus_name, 'q_set'] / args['phase_factor']
        baseKV = bus.loc[bus_name, 'v_nom']
        Gs = 0 # Gs = shunt_elem.loc[bus_name]
        Bs = 0 # Bs = shunt_elem.loc[bus_name]
        area = 0#int(bus_name[-1]) 
        zone = 0#country_code.index(bus_name[0:2]) 
        Vmax = args['voltage_upper']
        Vmin = args['voltage_lower']
        Vm = 1
        Va = 0
        busx = bus.loc[bus_name, 'x']
        busy = bus.loc[bus_name, 'y']
        Bus_data.append([bus_index, bus_type, Pd, Qd, Gs, Bs, area, Vm, Va, baseKV, zone, Vmax, Vmin, busx, busy])

    """
    Branch data
    fbus	tbus	r	x	b	rateA	rateB	rateC	ratio	angle	status	angmin	angmax num_line, num_seg, num_length
    Segment data
    fbus coor, inter node1, inter node2, inter node3 ,tbus coor
    """
    Branch_data = []
    branch = network.lines
    branch_id = branch.index.to_list()
    line_type = network.line_types
    segment_len = 25
    for i, branch_name in enumerate(branch_id):
        fbus_name = branch.loc[branch_name, 'bus0']
        tbus_name = branch.loc[branch_name, 'bus1']
        fbus_index = network.buses.index.tolist().index(fbus_name) 
        tbus_index = network.buses.index.tolist().index(tbus_name) 
        conductor_type = branch.loc[branch_name, 'type']
        # num_bundle = int(re.findall(r"(\d+)-bundle", conductor_type)[0])
        line_length = branch.loc[branch_name, 'length'] # km
        num_line = branch.loc[branch_name, 'num_parallel']
        num_seg = int(line_length // segment_len + 1)
        conductor_info = line_type.loc[conductor_type]
        unit_x = conductor_info['x_per_length'] # Ohm per km
        unit_r = conductor_info['r_per_length'] # Ohm per km
        unit_c = conductor_info['c_per_length'] # Shunt Capacitance nF per km (1 nF = 1e-9 F)
        f_nom = conductor_info['f_nom']
        i_nom = conductor_info['i_nom']
        v_nom = bus.loc[fbus_name, 'v_nom']

        BaseZ = (v_nom**2)/baseMVA # Om
        BaseY =  1/BaseZ # Siemens  

        s_nom = branch.loc[branch_name, 's_nom'] / phase_factor
        if args['safe_threshold']: # 0.7 in pypsa
            s_nom_max = branch.loc[branch_name, 's_max_pu']  # per-unit [0, 1]
        else:
            s_nom_max = 1

        rateA = s_nom * s_nom_max
        rateB = s_nom * s_nom_max
        rateC = s_nom * s_nom_max
        ratio = 0
        angle = 0
        if rateA == 0:
            status = 0
            Br_R = Br_X = 9999
            Br_B = 0
            Branch_data.append([fbus_index, tbus_index,	Br_R, Br_X,	Br_B, 
                                0, 0, 0, 
                                ratio, angle, status, 
                                -9999, 9999, 
                                num_line, num_seg, line_length])
        else:
            status = 1
            Br_R = unit_r * line_length / num_line  / BaseZ 
            Br_X = unit_x * line_length / num_line  / BaseZ 
            Br_B = 2 * np.pi * f_nom * unit_c * line_length * num_line * 1e-9 / BaseY 
            angmin = args['phase_angle_lower']
            angmax = args['phase_angle_upper']
            Branch_data.append([fbus_index, tbus_index,	Br_R, Br_X,	Br_B, 
                                rateA, rateB, rateC, 
                                ratio, angle, status, angmin, angmax, 
                                num_line, num_seg, line_length])
    Segment_data = network.segments

    

    """
    DC-Line data:
    fbus    tbus    status    Pf    Pt    Qf    Qt    Vf    Vt    Pmin    Pmax    QminF    QmaxF    QminT    QmaxT    loss0    loss1
    DC line cost data
    1    startup    shutdown    n    x1    y1    ...    xn    yn
    2    startup    shutdown    n    c(n-1)    ...    c0
    """
    DC_line_data = []
    DC_line_cost_data = []
    dcline = network.links
    dcline_id = dcline.index.to_list()
    for i, dcline_name in enumerate(dcline_id):
        fbus_name = dcline.loc[dcline_name, 'bus0']
        tbus_name = dcline.loc[dcline_name, 'bus1']    
        fbus_index = network.buses.index.tolist().index(fbus_name) 
        tbus_index = network.buses.index.tolist().index(tbus_name) 
        status = 1 - int(dcline.loc[dcline_name, 'under_construction'])
        p_nom = dcline.loc[dcline_name, 'p_nom'] / phase_factor
        efficiency = dcline.loc[dcline_name, 'efficiency']
        Pf = Pt = 0
        Qf = Qt = 0
        Vf = 1
        Vt = 1
        Pmin = dcline.loc[dcline_name, 'p_min_pu'] * p_nom
        Pmax = dcline.loc[dcline_name, 'p_max_pu'] * p_nom
        QminF = QmaxF = QminT =  QmaxT = 0
        loss0 = loss1 = 0
        capital_cost = dcline.loc[dcline_name, 'capital_cost']
        marginal_cost = dcline.loc[dcline_name, 'marginal_cost']
        marginal_cost_quadratic = 0#dcline.loc[dcline_name, 'marginal_cost_quadratic']

        DC_line_data.append([fbus_index, tbus_index, status, Pf, Pt, Qf,Qt, Vf, Vt, Pmin, Pmax, QminF, QmaxF, QminT, QmaxT, loss0, loss1])
        DC_line_cost_data.append([2, 0, 0, 3, marginal_cost_quadratic, marginal_cost, capital_cost])
        ## adding dcline-based branch to avoid network dis-connectness
        # Branch_data.append([fbus_index, tbus_index, 9999,9999,0, 0,0,0, 0,0,1,-9999,9999])


    baseKV = network.buses['v_nom'][0]
    baseI = baseMVA / baseKV
    ppc = {
        "version": '2',
        "baseMVA": baseMVA,
        'baseKV': baseKV,
        'baseI': baseI,
        "bus": np.array(Bus_data),
        "branch": np.array(Branch_data),
        "segment": Segment_data,
        "dcline": np.array(DC_line_data),
        "dclinecost": np.array(DC_line_cost_data),
        "gen": np.array(Gen_data),
        "gencost": np.array(Gen_cost_data),
        'pop_ratio': pop_ratio,
    }
    return ppc

def remove_isolated_elements(ppc):
    """
    Removes isolated buses, generators, and branches from the PYPOWER case file.
    :param ppc: The PYPOWER case dictionary.
    :return: The cleaned PYPOWER case dictionary.
    """
    # Step 1: Identify all connected buses
    connected_buses_idx = []
    for branch in ppc['branch']:
        if branch[idx_brch.BR_STATUS] == 1:  # branch status 1 means in-service
            connected_buses_idx.append(branch[0])
            connected_buses_idx.append(branch[1])
    for dcline in ppc['dcline']:
        if dcline[2] == 1:
            connected_buses_idx.append(dcline[0])
            connected_buses_idx.append(dcline[1])


    # step 2: Filter out isolated bus:
    new_bus_idx = [i for i, bus in enumerate(ppc['bus']) if bus[0] in  connected_buses_idx]
    ppc['bus'] = ppc['bus'][new_bus_idx]
    ppc['bus_id'] = ppc['bus_id'][new_bus_idx]
    # ppc['load'] = ppc['load'][:, new_bus_idx,:]
    ppc['pop_ratio'] = ppc['pop_ratio'][new_bus_idx]

    # Step 3: Filter out generators on isolated buses
    new_gen_idx = [i for i, gen in enumerate(ppc['gen']) if gen[0] in connected_buses_idx]
    ppc['gen'] = ppc['gen'][new_gen_idx]
    ppc['gencost'] = ppc['gencost'][new_gen_idx]
    # ppc['renewable'] = ppc['renewable'][:,new_gen_idx]

    # Step 4: Filter out branches with isolated buses
    new_branch_idx = [i for i, branch in enumerate(ppc['branch']) if branch[0] in connected_buses_idx and branch[1] in connected_buses_idx]
    ppc['branch'] = ppc['branch'][new_branch_idx]
    ppc['segment'] = ppc['segment'][new_branch_idx]
    # Step 5: Filter out dclines with isolated buses
    new_dcline_idx = [i for i, dcline in enumerate(ppc['dcline']) if dcline[0] in connected_buses_idx and dcline[1] in connected_buses_idx]
    ppc['dcline'] = ppc['dcline'][new_dcline_idx]
    ppc['dclinecost'] = ppc['dclinecost'][new_dcline_idx]

    # Step 5: Create a mapping from old bus indices to new bus indices
    old_bus_indices = ppc['bus'][:, 0]
    new_bus_indices = np.arange(len(old_bus_indices)) 
    bus_map = {old: new for old, new in zip(old_bus_indices, new_bus_indices)}

    # Reindex the buses
    ppc['bus'][:, 0] = new_bus_indices
    # Update the generator bus indices
    for gen in ppc['gen']:
        gen[0] = bus_map[gen[0]]
    # Update the branch bus indices
    for branch in ppc['branch']:
        branch[0] = bus_map[branch[0]]
        branch[1] = bus_map[branch[1]]
    for dcline in ppc['dcline']:
        dcline[0] = bus_map[dcline[0]]
        dcline[1] = bus_map[dcline[1]]
    return ppc

def check_connectiveness(ppc):
    G = nx.Graph()
    for i, bus in enumerate(ppc['bus']):
        G.add_node(i, pos=(bus[-2],bus[-1]))
    G.add_weighted_edges_from([(int(line[0]), int(line[1]), 1) for line in ppc['branch'] if line[idx_brch.BR_STATUS]==1])
    if nx.is_connected(G):
        print('Network Connected without DCLine')
        return True
    else:
        G.add_weighted_edges_from([(int(line[0]), int(line[1]), 1) for line in ppc['dcline'] if line[2]==1])
        if nx.is_connected(G):
            print('Network Connected with DCLine')
            return True
        else:
            print('Network Dis-Connected')
            # pos = nx.get_node_attributes(G, 'pos')
            # nx.draw(G, pos, node_size=2)
            # plt.show()
            return False
            # raise Exception('non-fully-connected network')

