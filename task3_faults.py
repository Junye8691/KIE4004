import pandapower as pp
import pandapower.networks as nw
import pandapower.shortcircuit as sc
import pandapower.topology as top
import networkx as nx
import numpy as np
import pandas as pd
import os
from pandapower.converter.matpower import from_mpc 


# --- NETWORK LOADER ---
def get_network(sys_name):
    print(f"Loading system: {sys_name}...")

    # Built-in IEEE 33-bus
    if sys_name == 'case33':
        return nw.case33bw()

    # MATPOWER IEEE 69-bus
    elif sys_name == 'case69':
        base_path = os.path.dirname(os.path.abspath(__file__))
        m_file_path = os.path.join(base_path, 'matpower', 'data', 'case69.m')

        if not os.path.exists(m_file_path):
            raise FileNotFoundError(f"Missing file: {m_file_path}")

        net = from_mpc(m_file_path, f_hz=50)

        # ---- UNIT CORRECTION (CRITICAL) ----
        # Load: kW → MW
        net.load.p_mw /= 1000.0
        net.load.q_mvar /= 1000.0

        # Line impedance correction
        base_mva = 10.0
        base_kv = 12.66
        z_base = (base_kv ** 2) / base_mva

        net.line.r_ohm_per_km /= z_base
        net.line.x_ohm_per_km /= z_base

        return net

    else:
        raise ValueError("Unsupported system name")


# --- NETWORK SETUP ---
def setup_network(sys_name):
    """
    Sets up the IEEE system with necessary Zero Sequence parameters.
    """
    net = get_network(sys_name)
    
    # --- DEFINE ZERO SEQUENCE PARAMETERS ---
    # Standard IEEE cases often lack Z0 data.
    net.ext_grid['s_sc_max_mva'] = 1000.0   # short circuit capacity of grid
    net.ext_grid['rx_max'] = 0.1            # R/X ratio of source
    net.ext_grid['x0x_max'] = 1.0           # Z0_grid = Z1_grid
    net.ext_grid['r0x0_max'] = 0.1          

    # Add Zero Sequence data to lines (Assumption: Z0 ≈ 3Z1 for overhead lines)
    net.line['r0_ohm_per_km'] = net.line['r_ohm_per_km'] * 3
    net.line['x0_ohm_per_km'] = net.line['x_ohm_per_km'] * 3
    net.line['c0_nf_per_km'] = net.line['c_nf_per_km']
    net.line['end_of_line'] = False 

    return net


# --- MANUAL THEVENIN IMPEDANCE ---
def get_thevenin_impedance_manual(net, fault_bus_idx):
    """
    Manually sums impedance from Source (Ext Grid) to Fault Bus.
    """
    # --- 1. External Grid Impedance ---
    grid_idx = 0 
    grid_bus_idx = net.ext_grid.at[grid_idx, 'bus']
    
    vn_kv = net.bus.at[grid_bus_idx, 'vn_kv'] 
    s_sc = net.ext_grid.at[grid_idx, 's_sc_max_mva']
    rx_ratio = net.ext_grid.at[grid_idx, 'rx_max']
    c = 1.1  # IEC 60909 Voltage Factor

    z_grid_mag = (c * vn_kv**2) / s_sc
    x_grid = z_grid_mag / np.sqrt(1 + rx_ratio**2)
    r_grid = x_grid * rx_ratio

    Z_grid_1 = complex(r_grid, x_grid)
    Z_grid_0 = Z_grid_1

    # --- 2. Find path from source to fault bus ---
    graph = top.create_nxgraph(net, respect_switches=False)
    path_buses = nx.shortest_path(graph, grid_bus_idx, fault_bus_idx)
    
    # --- 3. Sum line impedances ---
    Z_line_sum_1 = 0j
    Z_line_sum_0 = 0j
    
    for i in range(len(path_buses) - 1):
        from_b = path_buses[i]
        to_b = path_buses[i+1]
        
        line = net.line[((net.line.from_bus == from_b) & (net.line.to_bus == to_b)) |
                        ((net.line.from_bus == to_b) & (net.line.to_bus == from_b))]
        
        idx = line.index[0]
        length = net.line.at[idx, 'length_km']
        
        Z_line_sum_1 += complex(
            net.line.at[idx, 'r_ohm_per_km'] * length,
            net.line.at[idx, 'x_ohm_per_km'] * length
        )
        
        Z_line_sum_0 += complex(
            net.line.at[idx, 'r0_ohm_per_km'] * length,
            net.line.at[idx, 'x0_ohm_per_km'] * length
        )

    Z_1_total = Z_grid_1 + Z_line_sum_1
    Z_2_total = Z_1_total
    Z_0_total = Z_grid_0 + Z_line_sum_0
    
    return Z_0_total, Z_1_total, Z_2_total


# --- MANUAL FAULT CALCULATION ---
def calculate_manual_faults(net, fault_bus_idx, Z_f=0):
    """
    Performs analytical calculations using Symmetrical Components.
    """
    c = 1.1 
    vn_kv = net.bus.at[fault_bus_idx, 'vn_kv']
    E_a = (c * vn_kv * 1000) / np.sqrt(3)    # --> how to calculate E_a?
    
    Z_0, Z_1, Z_2 = get_thevenin_impedance_manual(net, fault_bus_idx)
    
    # Here correct
    a = -0.5 + 1j * (np.sqrt(3)/2)                  # eq (10.2) - L8
    A_matrix = np.array([[1, 1, 1],                 # eq (10.10) - L8
                         [1, a**2, a], 
                         [1, a, a**2]])

    results = {}

    # --- SLG ---
    I0 = E_a / (Z_0 + Z_1 + Z_2 + 3*Z_f)            # eq (10.62) - L9
    Iabc = A_matrix @ np.array([I0, I0, I0])        # eq (10.8) & eq (10.58)
    results['LG'] = abs(Iabc[0]) / 1000             # in kA --> check a bit

    # --- LL ---
    I1 = E_a / (Z_1 + Z_2 + Z_f)                    # eq (10.75) - L9  
    Iabc = A_matrix @ np.array([0, I1, -I1])        # eq (10.76) - L9
    results['LL'] = abs(Iabc[1]) / 1000

    # --- LLG ---
    Z_eq_0 = Z_0 + 3*Z_f                                    # Optional Definition
    I1 = E_a / (Z_1 + (Z_2 * Z_eq_0) / (Z_2 + Z_eq_0))      # eq (10.88) - L9
    I0 = -(E_a - Z_1 * I1) / Z_eq_0                         # eq (10.86) - L9
    results['LLG'] = abs(3 * I0) / 1000                     # eq (10.89) - L9

    return results


# --- TASK 3 ENTRY POINT (USED BY run_project.py) ---
def run_fault_analysis(system="33", fault_bus=1, fault_type="LG"):
    """
    Task 3 – Fault Analysis and Validation
    fault_type: "LG", "LL", "LLG"
    """
    sys_map = {"33": "case33", "69": "case69"}
    net = setup_network(sys_map[system])

    print(f"\n--- TASK 3: FAULT ANALYSIS (IEEE {system}-Bus | Bus {fault_bus + 1}) ---")

    manual = calculate_manual_faults(net, fault_bus)
    I_manual = manual[fault_type]

    if fault_type == "LG":
        sc.calc_sc(net, fault='1ph', bus=fault_bus)
        I_pp = net.res_bus_sc.ikss_ka.at[fault_bus]

    elif fault_type == "LL":
        sc.calc_sc(net, fault='2ph', bus=fault_bus)
        I_pp = net.res_bus_sc.ikss_ka.at[fault_bus]

    else:  # LLG
        I_pp = np.nan

    print(f"Manual Calculation  : {I_manual:.4f} kA")
    print(f"Pandapower Result   : {I_pp if not np.isnan(I_pp) else 'N/A'}")
    if not np.isnan(I_pp):
        print(f"Difference          : {abs(I_manual - I_pp):.4f} kA")


# --- STANDALONE RUN ---
if __name__ == "__main__":
    run_fault_analysis(system="69", fault_bus=10, fault_type="LG")
