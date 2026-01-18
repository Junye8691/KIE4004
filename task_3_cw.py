import pandapower as pp
import pandapower.networks as pn
import pandapower.shortcircuit as sc
from pandapower.converter.matpower import from_mpc
import pandapower.plotting as plot # Added for network visualization
import pandapower.topology as top
import networkx as nx
import os

import numpy as np
import matplotlib.pyplot as plt

def save_figure(fig_name, task_folder):
    base_path = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_path, task_folder)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, fig_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[Saved] {save_path}")

def build_distribution_network(system="33"):
    """
    system: "33" or "69"
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    if system == "33":
        net = pn.case33bw()
        # m_file_path = os.path.join(base_path, 'matpower', 'data', 'case33bw.m')
    elif system == "69":
        m_file_path = os.path.join(base_path, 'matpower', 'data', 'case69.m')

        if not os.path.exists(m_file_path):
            raise FileNotFoundError(f"Missing file: {m_file_path}")
        
        net = from_mpc(m_file_path, f_hz=50)
    else:
        raise ValueError("Unsupported system. Use '33 or '69'.")

    return net

def plot_network(net, fault_bus=None):
    """
    Generates a plot of the network topology and highlights the fault bus if provided.
    """
    # print(f"Generating plot for IEEE {system_name}-bus system...")
    
    # Default color is blue for all buses
    bus_colors = ["b"] * len(net.bus)
    
    # If a fault bus is specified, change its color to red
    if fault_bus is not None and fault_bus in net.bus.index:
        bus_colors[fault_bus] = "r"
        # print(f"Highlighting Fault at Bus {fault_bus} in RED.")

    plot.simple_plot(net, 
                     show_plot=False, 
                     bus_size=1.2, 
                     line_width=1.0, 
                     bus_color=bus_colors)

def set_sc_parameters(net):
    # Positive-sequence SC parameters
    net.ext_grid["s_sc_max_mva"] = 1000
    net.ext_grid["s_sc_min_mva"] = 1000
    net.ext_grid["rx_max"] = 0.1
    net.ext_grid["rx_min"] = 0.1

    # Zero-sequence parameters (IEC 60909 assumption)
    net.ext_grid["x0x_max"] = 3.0
    net.ext_grid["x0x_min"] = 3.0
    net.ext_grid["r0x0_max"] = 0.1
    net.ext_grid["r0x0_min"] = 0.1

    # Line zero-sequence data
    for line in net.line.index:
        net.line.at[line, "r0_ohm_per_km"] = 3 * net.line.at[line, "r_ohm_per_km"]
        net.line.at[line, "x0_ohm_per_km"] = 3 * net.line.at[line, "x_ohm_per_km"]
        net.line.at[line, "c0_nf_per_km"] = 0

    return net

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
    ratio = 3.0             # Typical 3 to 5 for solid grounded systems  
    Z_grid_0 = ratio * Z_grid_1

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

def calc_sequence_currents(Ea, fault_type, Z0, Z1, Z2, Z_f=0):
    if fault_type == "LG":                             
        I = Ea / (Z0 + Z1 + Z2 + 3*Z_f)                 # eq (10.62) - L9
        return np.array([I, I, I])                      # eq (10.8) & eq (10.58)

    elif fault_type == "LL":                            
        I1 = Ea / (Z1 + Z2 + Z_f)                       # eq (10.75) - L9
        return np.array([0, I1, -I1])                   # eq (10.76) - L9

    elif fault_type == "LLG":
        Z_eq = Z0 + 3*Z_f                               # Optional Definition          
        I1 = Ea / (Z1 + (Z2 * Z_eq) / (Z2 + Z_eq))      # eq (10.88) - L9
        I2 = -(Ea - Z1 * I1) / Z2                        # eq (10.87) - L9
        I0 = -(Ea - I1 * Z1) / Z_eq                      # eq (10.86) - L9  
        return np.array([I0, I1, I2])                   # eq (10.8) - L8

def calc_sequence_voltages(Ea, Z0, Z1, Z2, I012):           
    V1 = Ea - Z1 * I012[1]
    V2 = -Z2 * I012[2]
    V0 = -Z0 * I012[0]
    return np.array([V0, V1, V2])                       # eq (10.54) - L9

def run_fault(net, fault_bus, fault_type):
    if fault_type == "LG":
        pp_fault = "1ph"
    elif fault_type == "LL":
        pp_fault = "2ph"
    elif fault_type == "LLG":
        print("DLG (2ph-g) fault not supported in pandapower. ")
        pp_fault = None
    else:
        raise ValueError("Invalid fault type")

    if pp_fault is not None:
        sc.calc_sc(net, fault=pp_fault, bus=fault_bus, case="max")
    return net

def run_fault_analysis(system, fault_bus, fault_type):
    net = build_distribution_network(system)
    # plot_network(net, fault_bus=fault_bus)
    # save_figure(
    # fig_name=f"network_fault_bus{fault_bus}_{system}.png",
    # task_folder="task3_results"
    # )
    # plt.show()
    net = set_sc_parameters(net)

    # ---------------- BASE VALUES ----------------
    c = 1.1                                         # IEC 60909 Voltage Factor
    S_base_MVA = 10.0                               # choose a clean base (typical for distribution)
    S_base = S_base_MVA * 1e6                       # VA

    vn_kv = net.bus.at[fault_bus, 'vn_kv']
    V_base_LL = vn_kv * 1e3                         # line-line volts in V
    V_base_phase = V_base_LL / np.sqrt(3)

    Z_base = V_base_LL**2 / S_base                  # ohm
    I_base = S_base / (np.sqrt(3) * V_base_LL)

    Ea = c * V_base_phase                           # volts

    net = run_fault(net, fault_bus, fault_type)

    # Sequence networks
    Z0, Z1, Z2 = get_thevenin_impedance_manual(net, fault_bus)

    a = np.exp(1j * 2 * np.pi / 3)                  # eq (10.2) - L8

    A = np.array([                                  # eq (10.10) - L8
        [1, 1, 1],
        [1, a**2, a],
        [1, a, a**2]
    ])

    A_inv = (1/3) * np.array([                      # eq (10.12) - L8
        [1, 1, 1],
        [1, a, a**2],
        [1, a**2, a]
    ])

    # Fault currents
    I012 = calc_sequence_currents(Ea, fault_type, Z0, Z1, Z2)
    Iabc = A @ I012                                                 # eq (10.9) - L8

    # Fault voltages
    V012 = calc_sequence_voltages(Ea, Z0, Z1, Z2, I012)
    Vabc = A @ V012                                                 # eq (10.17) - L8     

    # ---------------- PU CONVERSION ----------------
    I012_pu = I012 / I_base
    Iabc_pu = Iabc / I_base

    V012_pu = V012 / V_base_phase
    Vabc_pu = Vabc / V_base_phase

    print("\n========================================================= TASK 3 – FAULT ANALYSIS =========================================================")
    print(f"Fault Type : {FAULT_TYPE.upper()}")
    print(f"Fault Bus  : {FAULT_BUS}")

    print("\nSequence Impedances (Ohm)")
    print(f"Z0 = {Z0:.4f}, Z1 = {Z1:.4f}, Z2 = {Z2:.4f}")

    print("\nSequence Currents (A)\t|\tPhase Currents (A)\t|\tSequence Voltages (V)\t|\tPhase Voltages (V)")
    print(f"I0 = {I012[0]:.4f}\t|\tIa = {Iabc[0]:.4f}\t|\tV0 = {V012[0]:.4f}\t|\tVa = {Vabc[0]:.4f}")
    print(f"I1 = {I012[1]:.4f}\t|\tIb = {Iabc[1]:.4f}\t|\tV1 = {V012[1]:.4f}\t|\tVb = {Vabc[1]:.4f}")
    print(f"I2 = {I012[2]:.4f}\t|\tIc = {Iabc[2]:.4f}\t|\tV2 = {V012[2]:.4f}\t|\tVc = {Vabc[2]:.4f}")

    print("\nSequence Currents (pu)\t|\tPhase Currents (pu)\t|\tSequence Voltages (pu)\t|\tPhase Voltages (pu)")
    print(f"I0 = {I012_pu[0]:.4f}\t|\tIa = {Iabc_pu[0]:.4f}\t|\tV0 = {V012_pu[0]:.4f}\t|\tVa = {Vabc_pu[0]:.4f}")
    print(f"I1 = {I012_pu[1]:.4f}\t|\tIb = {Iabc_pu[1]:.4f}\t|\tV1 = {V012_pu[1]:.4f}\t|\tVb = {Vabc_pu[1]:.4f}")
    print(f"I2 = {I012_pu[2]:.4f}\t|\tIc = {Iabc_pu[2]:.4f}\t|\tV2 = {V012_pu[2]:.4f}\t|\tVc = {Vabc_pu[2]:.4f}")

    print("\nres_bus_sc columns:")
    print(net.res_bus_sc.columns)

    if fault_type == "LG" or fault_type == "LL":
        rk_pp  = net.res_bus_sc.at[fault_bus, "rk_ohm"]
        xk_pp  = net.res_bus_sc.at[fault_bus, "xk_ohm"]

        Z1_pp = complex(rk_pp,  xk_pp)
        print("\n--- Thevenin Impedance Comparison ---")
        print(f"Analytical Z1 = {Z1:.4f} ohm")
        print(f"Pandapower Z1 = {Z1_pp} ohm\n")

    if fault_type == "LG":
        rk0_pp = net.res_bus_sc.at[fault_bus, "rk0_ohm"]
        xk0_pp = net.res_bus_sc.at[fault_bus, "xk0_ohm"]

        Z0_pp = complex(rk0_pp, xk0_pp)
        print(f"Analytical Z0 = {Z0:.4f} ohm")
        print(f"Pandapower Z0 = {Z0_pp} ohm")

    if fault_type == "LG":
        Ia_analytical_ka = np.abs(Iabc[0]) / 1000
    elif fault_type == "LL":
        Ia_analytical_ka = np.abs(Iabc[1]) / 1000
    elif fault_type == "LLG":
        Ia_analytical_ka = np.abs(Iabc[1]+Iabc[2]) / 1000

    print("\n--- Fault Current Comparison ---")
    print(f"Analytical |Ia| = {Ia_analytical_ka:.4f} kA")

    if fault_type == "LG" or fault_type == "LL":
        ikss_pp = net.res_bus_sc.at[fault_bus, "ikss_ka"]
        print(f"Pandapower ikss = {ikss_pp} kA")
        print(f"Difference      = {abs(Ia_analytical_ka - ikss_pp):.4f} kA")

    print("===========================================================================================================================================")

# ================================= For individual task  ================================= 
FAULT_BUS = 11
FAULT_TYPE = "LG"   # Options: "LG", "LL", "LLG"
SYSTEM = "33"   # "33" or "69"

if __name__ == "__main__":
        run_fault_analysis(
        system=SYSTEM,
        fault_bus=FAULT_BUS,
        fault_type=FAULT_TYPE
    )

# Z_f = 0 pu (solid fault)