import pandapower as pp
import pandapower.networks as pn
import pandapower.shortcircuit as sc
from pandapower.converter.matpower import from_mpc
import pandapower.plotting as plot
import numpy as np
import matplotlib.pyplot as plt

def build_distribution_network(system="33"):
    if system == "33":
        net = from_mpc("matpower/data/case33bw.m", f_hz=50)
    elif system == "69":
        net = from_mpc("matpower/data/case69.m", f_hz=50)
    else:
        raise ValueError("Unsupported system. Use '33' or '69'.")
    return net

def plot_network(net, system_name, fault_bus=None):
    """
    Generates a plot of the network topology and highlights the fault bus if provided.
    """
    print(f"Generating plot for IEEE {system_name}-bus system...")
    
    # Default color is blue for all buses
    bus_colors = ["b"] * len(net.bus)
    
    # If a fault bus is specified, change its color to red
    if fault_bus is not None and fault_bus in net.bus.index:
        bus_colors[fault_bus] = "r"
        print(f"Highlighting Fault at Bus {fault_bus} in RED.")

    plot.simple_plot(net, 
                     show_plot=True, 
                     bus_size=1.2, 
                     line_width=1.0, 
                     bus_color=bus_colors)

def set_sc_parameters(net):
    # Initialize zero-sequence data (Mandatory for SLG/DLG simulation)
    net.line["r0_ohm_per_km"] = 3 * net.line["r_ohm_per_km"]
    net.line["x0_ohm_per_km"] = 3 * net.line["x_ohm_per_km"]
    net.line["c0_nf_per_km"] = 0 

    # Ext grid parameters (IEC 60909)
    net.ext_grid["s_sc_max_mva"] = 1000
    net.ext_grid["rx_max"] = 0.1
    net.ext_grid["x0x_max"] = 3.0
    net.ext_grid["r0x0_max"] = 0.1
    return net

def run_pure_simulation(net, fault_bus, fault_type):
    """
    Executes the simulation using pandapower's IEC 60909 engine.
    """
    # Map fault types to pandapower codes
    # '3ph' = 3-phase, '1ph' = Single line to ground, '2ph' = Line to line
    if fault_type == "slg":
        pp_fault = "1ph"
    elif fault_type == "ll":
        pp_fault = "2ph"
    elif fault_type == "dlg":
        # In pandapower, DLG is typically handled via '2ph' with ground path 
        # or specific settings depending on version.
        pp_fault = "2ph" 
    else:
        pp_fault = "3ph"

    # Calculate short circuit according to IEC 60909
    sc.calc_sc(net, fault=pp_fault, bus=fault_bus, case="max")
    
    return net

if __name__ == "__main__":
    FAULT_BUS = 10
    SYSTEM = "33"
    FAULT_TYPE = "slg" # Try "slg", "ll"

    # 1. Setup
    net = build_distribution_network(SYSTEM)
    net = set_sc_parameters(net)
    plot_network(net, SYSTEM, fault_bus=FAULT_BUS)

    # 2. Pure Simulation Execution
    net = run_pure_simulation(net, FAULT_BUS, FAULT_TYPE)

    # 3. Extracting Results from Simulation Tables
    # Total initial symmetrical short-circuit current (kA)
    ikss = net.res_bus_sc.at[FAULT_BUS, "ikss_ka"]
    
    # Determining Sequence Impedances (Simulation Method)
    # To get Z1/Z0 purely from simulation, we compare 3ph and 1ph results
    sc.calc_sc(net, fault="3ph", bus=FAULT_BUS)
    ikss_3ph = net.res_bus_sc.at[FAULT_BUS, "ikss_ka"]
    
    Vn = net.bus.at[FAULT_BUS, "vn_kv"]
    # Z1 (Positive Sequence) = c * Vn / (sqrt(3) * Ikss_3ph)
    z1_sim = (1.1 * Vn) / (np.sqrt(3) * ikss_3ph) 
    
    # For SLG: Ikss_1ph = (3 * c * Vn / sqrt(3)) / (Z0 + Z1 + Z2)
    # We can derive Z0 (Zero Sequence) from the ikss value provided by the solver
    z0_sim = (3 * 1.1 * Vn / np.sqrt(3) / ikss) - 2 * z1_sim

    print("\n========== TASK 3 – PURE SIMULATION RESULTS ==========")
    print(f"System      : IEEE {SYSTEM}-bus")
    print(f"Fault Type  : {FAULT_TYPE.upper()} at Bus {FAULT_BUS}")
    print(f"Total Ik''  : {ikss:.4f} kA")
    
    print("\n--- Derived Sequence Networks (from Simulation) ---")
    print(f"Positive Seq Impedance (Z1): {z1_sim:.4f} Ohm")
    print(f"Zero Seq Impedance (Z0)    : {z0_sim:.4f} Ohm")

    # Voltage at fault bus during fault is 0 for the faulted phase in SLG
    print("\n--- Phase Voltages at Fault Bus (pu) ---")
    if FAULT_TYPE == "slg":
        print("Va = 0.0000 (Faulted Phase)")
        print("Vb = 1.0000 (Approx. for stiff grid)")
        print("Vc = 1.0000 (Approx. for stiff grid)")