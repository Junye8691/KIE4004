import pandapower as pp
import pandapower.networks as nw
from pandapower.converter.matpower import from_mpc 
import time
import pandas as pd
import os

# --- CONFIGURATION ---
TARGET_SYSTEM = 'case69' 

def get_network(sys_name):
    print(f"Loading system: {sys_name}...")
    
    if hasattr(nw, sys_name):
        return getattr(nw, sys_name)()

    base_path = os.path.dirname(os.path.abspath(__file__)) 
    m_file_path = os.path.join(base_path, 'matpower', 'data', f'{sys_name}.m')

    if not os.path.exists(m_file_path):
        raise FileNotFoundError(f"File missing at: {m_file_path}")
    
    print(f"   -> Found local file. Converting: {m_file_path}")
    net = from_mpc(m_file_path, f_hz=50)

    # === FIX: SCALE UNITS ===
    if sys_name == 'case69':
        print("   -> Applying Unit Correction (kW->MW, Ohms->Actual)...")
        
        # Fix Loads: Divide by 1000 to convert kW to MW
        net.load.p_mw /= 1000.0
        net.load.q_mvar /= 1000.0
        
        # Fix Lines: Undo the incorrect p.u. conversion done by from_mpc
        # Z_base = (12.66^2) / 10 = 16.027 Ohms
        base_mva = 10.0
        base_kv = 12.66
        z_base = (base_kv**2) / base_mva
        
        net.line.r_ohm_per_km /= z_base
        net.line.x_ohm_per_km /= z_base

    return net

def run_powerflow(net, algo_name, algo_code):
    pp.reset_results(net)
    t0 = time.time()
    try:
        # Use Standard Newton-Raphson
        pp.runpp(net, algorithm=algo_code)
        
        exec_time = time.time() - t0
        try:
            iterations = net._ppc['iterations']
        except:
            iterations = "N/A"

        total_loss_mw = net.res_line.pl_mw.sum()
        min_voltage_pu = net.res_bus.vm_pu.min()
        
        return [f"{exec_time:.5f}", iterations, f"{total_loss_mw:.4f}", f"{min_voltage_pu:.4f}"]

    except pp.LoadflowNotConverged:
        return ["DNC", "Max", "-", "-"]
    except Exception as e:
        print(f"Error: {e}")
        return ["Fail", "-", "-", "-"]

if __name__ == "__main__":
    net = get_network(TARGET_SYSTEM)
    print(f"   -> System Loaded. Buses: {len(net.bus)}")

    # Ensure Slack Bus exists
    if len(net.ext_grid) == 0:
        print("   -> Adding Missing Slack Bus at index 0...")
        pp.create_ext_grid(net, bus=0, vm_pu=1.0)

    print("\n--- Starting Simulation ---")
    nr_results = run_powerflow(net, "Newton-Raphson", 'nr')
    fd_results = run_powerflow(net, "Fast Decoupled", 'fdbx')

    df = pd.DataFrame({
        'Metric': ['Time (s)', 'Iterations', 'Total Loss (MW)', 'Min Voltage (p.u.)'],
        'Newton-Raphson': nr_results,
        'Fast Decoupled': fd_results
    })
    
    print("\n" + "="*50)
    print(f"PERFORMANCE COMPARISON: {TARGET_SYSTEM}")
    print("="*50)
    print(df.to_string(index=False))
    print("="*50)