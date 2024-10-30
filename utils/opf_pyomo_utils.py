import os
import numpy as np
import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.contrib.pynumero.interfaces.pyomo_nlp import PyomoNLP
from pyomo.util.infeasible import log_infeasible_constraints

from pypower.api import case30, ppoption, runpf, loadcase
from pypower.makeYbus import makeYbus
from pypower.ext2int import ext2int
from pypower.makeBdc import makeBdc
from pypower.api import ppoption, runopf, rundcopf, runpf, makeYbus, makeBdc
from pypower import idx_bus, idx_gen, idx_brch
from scipy.io import loadmat
from utils.heat_flow_utils import coefficient_quadratic_approximation

def pyomo_solve_ac(ppc, solver_name='ipopt', tol=1e-3, 
                   ex_gen=False, initial_value=None, tem_cons=False, 
                   qua_con=False, angle_cons=False, qlim=False,
                   weather=None, conductor=None):
    if 'dcline' not in ppc:
        ppc['dcline'] = []
    bus, gen, gencost, branch, dcline = ppc["bus"], ppc["gen"], ppc["gencost"], ppc["branch"], ppc['dcline']
    baseMVA = ppc['baseMVA']
    baseKV = bus[0, idx_bus.BASE_KV]
    BaseI = baseMVA / baseKV

    nbus = len(bus)
    ngen = len(gen)
    nbranch = len(branch)
    ndcline = len(dcline)
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
    num_bundle = ppc['branch'][:, -3] 
    num_bundle[num_bundle==0] = 1
    Branch_status = np.sign(ppc['branch'][:, idx_brch.BR_STATUS] * ppc['branch'][:, idx_brch.RATE_A])
    gen_mask = np.array([[1 if gen[j, 0] == i else 0.0 for j in np.arange(ngen) ] for i in range(nbus)])
    dcf_mask = np.array([[1  if dcline[j, 0] == i else 0.0 for j in np.arange(ndcline)] for i in range(nbus)])
    dct_mask = np.array([[1  if dcline[j, 1] == i else 0.0  for j in np.arange(ndcline)] for i in range(nbus)])    
    fbus = branch[:, 0].astype(int)
    tbus = branch[:, 1].astype(int)
    if qua_con:
        Tmax = conductor['max_temperature']
        seg_prop = ppc['segment'][:, :, 2]
        beta0, beta1, beta2 = coefficient_quadratic_approximation(conductor, weather)


    ### --------------------- Pyomo model ------------------------------
    # Define the model
    model = pyo.ConcreteModel()
    # Define sets
    model.buses = pyo.Set(initialize=bus[:, 0].astype(int).tolist())
    model.ac_lines = pyo.Set(initialize=np.arange(len(branch)))
    model.dc_lines = pyo.Set(initialize=np.arange(len(dcline)))
    model.generators = pyo.Set(initialize=np.arange(len(gen)))
    # Define variables
    if initial_value is None:
        model.VM = pyo.Var(model.buses, domain = pyo.Reals, initialize=1.0)
        model.VA = pyo.Var(model.buses, domain = pyo.Reals, initialize=0.0)
        model.PG = pyo.Var(model.generators, domain = pyo.Reals, initialize=0)
        model.QG = pyo.Var(model.generators, domain = pyo.Reals, initialize=0)
        model.PDC = pyo.Var(model.dc_lines, domain = pyo.Reals, initialize=0.0)
    else:
        model.VM = pyo.Var(model.buses, domain = pyo.Reals, initialize=initial_value['VM'])
        model.VA = pyo.Var(model.buses, domain = pyo.Reals, initialize=initial_value['VA'])
        model.PG = pyo.Var(model.generators, domain = pyo.Reals, initialize=initial_value['PG'])
        model.QG = pyo.Var(model.generators, domain = pyo.Reals, initialize=initial_value['QG'])
        model.PDC = pyo.Var(model.dc_lines, domain = pyo.Reals, initialize=initial_value['PDC'])
    model.LS = pyo.Var(model.buses, domain = pyo.Reals, initialize=0.0)
    VR = [model.VM[i] * pyo.cos(model.VA[i])  for i in model.buses]
    VI = [model.VM[i] * pyo.sin(model.VA[i])  for i in model.buses]

    """
    Define objective function
    """
    obj = sum(model.PG[i] * gencost[i][5] for i in model.generators) + \
          sum(model.LS[i] * 1e7 for i in model.buses)
    model.objective = pyo.Objective(expr=obj, sense=pyo.minimize)


    """
    Add operational bound (gen, vol, dcline...)
    """
    for i in range(len(bus)):
        model.VM[i].lb = bus[i, idx_bus.VMIN]
        model.VM[i].ub = bus[i, idx_bus.VMAX]
        model.LS[i].lb = 0
        if bus[i][idx_bus.BUS_TYPE] == 3:
            model.VA[i].fix(0.0)

    for i in range(len(gen)):
        if gen[i, idx_gen.GEN_STATUS] == 1:
            model.PG[i].lb =  gen[i, idx_gen.PMIN] / baseMVA
            model.PG[i].ub =  gen[i, idx_gen.PMAX] / baseMVA
            if qlim:
                model.QG[i].lb =  gen[i, idx_gen.QMIN] / baseMVA
                model.QG[i].ub =  gen[i, idx_gen.QMAX] / baseMVA
        else:
            model.PG[i].fix(0.0)
            model.QG[i].fix(0.0)

    for i in range(len(dcline)):
        if dcline[i, 2] == 1:
            model.PDC[i].lb = dcline[i, 9] / baseMVA
            model.PDC[i].ub = dcline[i, 10] / baseMVA
        else:
             model.PDC[i].fix(0.0)

    """
    Add constraint in bus (power balance)
    """
    model.real_power_balance_constraint = pyo.ConstraintList()
    model.imag_power_balance_constraint = pyo.ConstraintList()
    epsilon = np.float64(0)
    for i in range(len(bus)):
        ### active power balance
        bus_real_injection = sum(model.PG[j] * gen_mask[i,j] for j in model.generators) \
                            - sum(model.PDC[j] * dcf_mask[i,j] for j in model.dc_lines) \
                            + sum(model.PDC[j] * dct_mask[i,j] for j in model.dc_lines) \
                            + model.LS[i] \
                            - bus[i, idx_bus.PD] / baseMVA
        ### reactive power balance
        bus_imag_injection = sum(model.QG[j] * gen_mask[i,j] for j in model.generators) \
                            - bus[i, idx_bus.QD] / baseMVA
        
        I_real = sum((Ybus[i, k].real * VR[k] - Ybus[i, k].imag * VI[k]) for k in model.buses)
        I_imag = sum((Ybus[i, k].real * VI[k] + Ybus[i, k].imag * VR[k]) for k in model.buses)
        bus_real_flow = VR[i] * I_real + VI[i] * I_imag
        bus_imag_flow = VI[i] * I_real - VR[i] * I_imag    
        # bus_real_flow = model.VM[i] * sum(model.VM[j] * (Ybus[i, j].real * pyo.cos(model.VA[i] - model.VA[j]) + \
        #                                                 Ybus[i, j].imag * pyo.sin(model.VA[i] - model.VA[j])) for j in model.buses)
        # bus_imag_flow = model.VM[i] * sum(model.VM[j] * (Ybus[i, j].real * pyo.sin(model.VA[i] - model.VA[j]) - \
        #                                                 Ybus[i, j].imag * pyo.cos(model.VA[i] - model.VA[j])) for j in model.buses)
        if isinstance(bus_real_injection+epsilon, np.float64) and isinstance(bus_real_flow+epsilon, np.float64):
            pass
        else:
            model.real_power_balance_constraint.add(bus_real_injection == bus_real_flow)
        if isinstance(bus_imag_injection+epsilon, np.float64) and isinstance(bus_imag_flow+epsilon, np.float64):
            pass
        else:
            model.imag_power_balance_constraint.add(bus_imag_injection == bus_imag_flow)

    """
    Add constraint in branch (flow, current, temperature, angle)
    """
    model.branch_flow_constraint = pyo.ConstraintList()
    model.branch_thermal_constraint = pyo.ConstraintList()
    model.branch_angle_constraint = pyo.ConstraintList()
    for l in range(len(branch)):
        if branch[l, idx_brch.BR_STATUS] == 1:
            i, j = int(branch[l, 0]), int(branch[l, 1])
            flow_limit_2 = (branch[l, idx_brch.RATE_A] / baseMVA)**2
            ### from branch flow constraint
            # real_power_flow_ij = model.VM[i] * sum( model.VM[k] * (Yf[l, k].real * pyo.cos(model.VA[i] - model.VA[k]) + 
            #                                                     Yf[l, k].imag * pyo.sin(model.VA[i] - model.VA[k])) for k in model.buses)
            # imag_power_flow_ij = model.VM[i] * sum( model.VM[k] * (Yf[l, k].real * pyo.sin(model.VA[i] - model.VA[k]) - 
            #                                                     Yf[l, k].imag * pyo.cos(model.VA[i] - model.VA[k])) for k in model.buses)
            # ### to branch flow constraint
            # real_power_flow_ji = model.VM[j] * sum( model.VM[k] * (Yt[l, k].real * pyo.cos(model.VA[j] - model.VA[k]) + 
            #                                                     Yt[l, k].imag * pyo.sin(model.VA[j] - model.VA[k])) for k in model.buses)
            # imag_power_flow_ji = model.VM[j] * sum( model.VM[k] * (Yt[l, k].real * pyo.sin(model.VA[j] - model.VA[k]) - 
            #                                                     Yt[l, k].imag * pyo.cos(model.VA[j] - model.VA[k])) for k in model.buses)
            # model.branch_flow_constraint.add(real_power_flow_ij**2 + imag_power_flow_ij**2 <= flow_limit_2)
            # model.branch_flow_constraint.add(real_power_flow_ji**2 + imag_power_flow_ji**2 <= flow_limit_2)
            
            If_real = sum((Yf[l, k].real * VR[k] - Yf[l, k].imag * VI[k]) for k in model.buses)
            If_imag = sum((Yf[l, k].real * VI[k] + Yf[l, k].imag * VR[k]) for k in model.buses)
            It_real = sum((Yt[l, k].real * VR[k] - Yt[l, k].imag * VI[k]) for k in model.buses)
            It_imag = sum((Yt[l, k].real * VI[k] + Yt[l, k].imag * VR[k]) for k in model.buses)

            real_power_flow_ij = VR[i] * If_real + VI[i] * If_imag
            imag_power_flow_ij = VI[i] * If_real - VR[i] * If_imag    
            real_power_flow_ji = VR[j] * It_real + VI[j] * It_imag
            imag_power_flow_ji = VI[j] * It_real - VR[j] * It_imag     

            model.branch_flow_constraint.add(real_power_flow_ij**2 + imag_power_flow_ij**2 <= flow_limit_2)
            model.branch_flow_constraint.add(real_power_flow_ji**2 + imag_power_flow_ji**2 <= flow_limit_2)

            if qua_con:
                If2 = (If_real**2 + If_imag**2) * (BaseI * 1000 / num_bundle[l] * Branch_status[l]) ** 2
                It2 = (It_real**2 + It_imag**2) * (BaseI * 1000 / num_bundle[l] * Branch_status[l]) ** 2
                if beta0.shape[1] == 1:
                    model.branch_thermal_constraint.add(beta0[l] 
                                                            + beta1[l] * If2
                                                            + beta2[l] * If2 **2 <= Tmax)
                    model.branch_thermal_constraint.add(beta0[l] 
                                                            + beta1[l] * It2
                                                            + beta2[l] * It2 **2 <= Tmax)
                else:
                    for s in range(seg_prop.shape[1]):
                        if seg_prop[l,s]>0:
                            model.branch_thermal_constraint.add(beta0[l,s] 
                                                                    + beta1[l,s] * If2
                                                                    + beta2[l,s] * If2 **2 <= Tmax)
                            model.branch_thermal_constraint.add(beta0[l,s] 
                                                                    + beta1[l,s] * It2
                                                                    + beta2[l,s] * It2 **2 <= Tmax)
            if tem_cons:
                current_limit_2 = (branch[l, idx_brch.RATE_B] / BaseI) ** 2
                model.branch_thermal_constraint.add(If_real**2 + If_imag**2 <= current_limit_2)
                model.branch_thermal_constraint.add(It_real**2 + It_imag**2 <= current_limit_2)
            if angle_cons:
                angle_max = (branch[l, idx_brch.ANGMAX])
                angle_min = (branch[l, idx_brch.ANGMIN])
                model.branch_angle_constraint.add(model.VA[i] - model.VA[j] <= angle_max / 180 * np.pi)
                model.branch_angle_constraint.add(model.VA[i] - model.VA[j] >= angle_min / 180 * np.pi)


    """
    Solve the optimization problem
    """ 
    solver = pyo.SolverFactory(solver_name)
    solver.options['tol'] = tol
    try:
        solver.solve(model, tee=False)
    except:
        log_infeasible_constraints(model)


    """
    Calulating statistics
    """ 
    # Pex = np.array([model.Pex[i].value for i in model.generators])
    PG = np.array([model.PG[i].value  for i in model.generators])
    QG = np.array([model.QG[i].value  for i in model.generators])
    VM = np.array([model.VM[i].value for i in model.buses])
    VA = np.array([model.VA[i].value for i in model.buses])
    PDC = np.array([model.PDC[i].value  for i in model.dc_lines])
    obj =  np.sum([PG[i]*baseMVA * gencost[i, 5] for i in np.arange(ngen)])  
    V = VM * np.exp(1j * VA)
    bus_real_injection = np.sum(gen_mask * PG , axis=1) \
                        - np.sum(dcf_mask * PDC, axis=1) \
                        + np.sum(dct_mask * PDC, axis=1) \
                        - bus[:, idx_bus.PD] / baseMVA
    bus_imag_injection = np.sum(gen_mask * QG, axis=1) \
                        - bus[:, idx_bus.QD] / baseMVA
    bus_flow = V * np.conj(Ybus @ V)
    If = Yf @ V
    It = Yt @ V
    Sf = V[fbus] * np.conj(If)
    St = V[tbus] * np.conj(It)
    S = np.maximum(np.abs(Sf), np.abs(St))
    I = np.maximum(np.abs(If), np.abs(It))
    mis_match = bus_real_injection + 1j * bus_imag_injection - bus_flow
    p_eq_vio = np.abs(np.real(mis_match))
    q_eq_vio = np.abs(np.imag(mis_match))
    eq_vio = np.abs(mis_match)
    ineq_vio_s = np.maximum(S - (branch[:, idx_brch.RATE_A] / baseMVA), 0)
    ineq_vio_i = np.maximum(I - (branch[:, idx_brch.RATE_B] / BaseI), 0)
    results = {'baseMVA': baseMVA, 'baseKV': baseKV, 'baseI': BaseI,
               'obj': obj, 'eq_vio': eq_vio, 'p_eq_vio': p_eq_vio, 'q_eq_vio': q_eq_vio,
               'ineq_vio_s': ineq_vio_s, 'ineq_vio_i': ineq_vio_i,
               'PD': bus[:, idx_bus.PD] / baseMVA, 'QD': bus[:, idx_bus.PD] / baseMVA,
               'Pex': None, 'PG': PG, 'QG': QG, 'VM': VM, 'VA': VA, 'PDC': PDC,
               'bus_flow': bus_flow, 'S_pu': S, 'I_pu': I}
    return results

