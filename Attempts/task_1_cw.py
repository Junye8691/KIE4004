import pandapower as pp
import pandapower.networks as nw
import time
import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
# Choose your system: 'case33bw', 'case118', or 'case69'
TARGET_SYSTEM = 'case69'
EXPORT_RESULTS = True  # Set to True to export results to CSV 

def get_network(sys_name):
    """Loads the appropriate IEEE test system"""
    if sys_name == 'case33bw':
        return nw.case33bw() # Standard IEEE 33 bus
    elif sys_name == 'case118':
        return nw.case118()  # Standard IEEE 118 bus
    elif sys_name == 'case69':
        # NOTE: IEEE 69 is not always built-in by default in older versions.
        # If this fails, we will load your .m file directly.
        try:
            return nw.case69() 
        except:
            # Fallback: Load from the MATPOWER .m file you downloaded
            # Update this path to where your case69.m is located
            return pp.converter.from_mpc(f'Z:/Power_System/assignment/matpower/data/{sys_name}.m')
    else:
        raise ValueError("Invalid System Selected")

print(f"=== Analyzing System: {TARGET_SYSTEM} ===")
net = get_network(TARGET_SYSTEM)

# Create output directory for results
output_dir = f"results_{TARGET_SYSTEM}"
if EXPORT_RESULTS and not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 1. Newton-Raphson (NR) Analysis ---
print("\nRunning Newton-Raphson...")
t0 = time.time()
try:
    # algorithm='nr' is Newton-Raphson
    pp.runpp(net, algorithm='nr')
    nr_time = time.time() - t0
    nr_iter = net._ppc['iterations'] # Internal iteration counter
    nr_loss = net.res_line.pl_mw.sum()
    nr_vmin = net.res_bus.vm_pu.min()
    nr_vmax = net.res_bus.vm_pu.max()
    
    # Store NR results for comparison
    nr_bus_vm = net.res_bus.vm_pu.copy()
    nr_bus_va = net.res_bus.va_degree.copy()
    nr_line_pl = net.res_line.pl_mw.copy()
    nr_line_ql = net.res_line.ql_mvar.copy()
    
    nr_success = True
except Exception as e:
    nr_success = False
    print(f"NR Failed: {e}")

# --- 2. Fast Decoupled (FDLF) Analysis ---
# Reset the net results first
pp.reset_results(net)
print("Running Fast Decoupled...")
t0 = time.time()
try:
    # algorithm='fdbx' is Fast Decoupled (XB variant)
    pp.runpp(net, algorithm='fdbx')
    fd_time = time.time() - t0
    fd_iter = net._ppc['iterations']
    fd_loss = net.res_line.pl_mw.sum()
    fd_vmin = net.res_bus.vm_pu.min()
    fd_vmax = net.res_bus.vm_pu.max()
    
    # Store FDLF results for comparison
    fd_bus_vm = net.res_bus.vm_pu.copy()
    fd_bus_va = net.res_bus.va_degree.copy()
    fd_line_pl = net.res_line.pl_mw.copy()
    fd_line_ql = net.res_line.ql_mvar.copy()
    
    fd_success = True
except Exception as e:
    fd_success = False
    print(f"FDLF Failed: {e}")

# --- 3. Comparison Table (Required for Report) ---
print("\n" + "="*60)
print("CONVERGENCE & EFFICIENCY COMPARISON")
print("="*60)

results = {
    'Metric': ['Time (s)', 'Iterations', 'Total Loss (MW)', 'Min Voltage (p.u.)', 'Max Voltage (p.u.)'],
    'Newton-Raphson': [f"{nr_time:.5f}", nr_iter, f"{nr_loss:.4f}", f"{nr_vmin:.4f}", f"{nr_vmax:.4f}"] if nr_success else ["Fail"]*5,
    'Fast Decoupled': [f"{fd_time:.5f}", fd_iter, f"{fd_loss:.4f}", f"{fd_vmin:.4f}", f"{fd_vmax:.4f}"] if fd_success else ["Fail"]*5
}

df_comparison = pd.DataFrame(results)
print(df_comparison.to_string(index=False))

# --- 4. Numerical Accuracy Comparison ---
if nr_success and fd_success:
    print("\n" + "="*60)
    print("NUMERICAL ACCURACY COMPARISON (NR vs FDLF)")
    print("="*60)
    
    # Voltage magnitude difference
    vm_diff = np.abs(nr_bus_vm - fd_bus_vm)
    vm_max_diff = vm_diff.max()
    vm_mean_diff = vm_diff.mean()
    
    # Voltage angle difference
    va_diff = np.abs(nr_bus_va - fd_bus_va)
    va_max_diff = va_diff.max()
    va_mean_diff = va_diff.mean()
    
    # Line loss difference
    pl_diff = np.abs(nr_line_pl - fd_line_pl)
    pl_max_diff = pl_diff.max()
    pl_mean_diff = pl_diff.mean()
    
    accuracy_results = {
        'Parameter': ['Max Voltage Magnitude Diff (p.u.)', 'Mean Voltage Magnitude Diff (p.u.)', 
                      'Max Voltage Angle Diff (deg)', 'Mean Voltage Angle Diff (deg)',
                      'Max Line Loss Diff (MW)', 'Mean Line Loss Diff (MW)'],
        'Value': [f"{vm_max_diff:.6f}", f"{vm_mean_diff:.6f}", 
                  f"{va_max_diff:.4f}", f"{va_mean_diff:.4f}",
                  f"{pl_max_diff:.6f}", f"{pl_mean_diff:.6f}"]
    }
    
    df_accuracy = pd.DataFrame(accuracy_results)
    print(df_accuracy.to_string(index=False))

# --- 5. BASE CASE RESULTS (Using Newton-Raphson as reference) ---
if nr_success:
    # Re-run NR to ensure results are loaded
    pp.runpp(net, algorithm='nr')
    
    print("\n" + "="*60)
    print("BASE CASE RESULTS - BUS VOLTAGES")
    print("="*60)
    
    # Bus voltage results
    bus_results = pd.DataFrame({
        'Bus': net.bus.index,
        'Name': net.bus.name if 'name' in net.bus.columns else ['Bus' + str(i) for i in net.bus.index],
        'V_magnitude (p.u.)': net.res_bus.vm_pu,
        'V_angle (deg)': net.res_bus.va_degree,
        'P_load (MW)': net.res_bus.p_mw,
        'Q_load (MVAr)': net.res_bus.q_mvar
    })
    print(bus_results.to_string(index=False))
    
    print("\n" + "="*60)
    print("BASE CASE RESULTS - LINE POWER LOSSES")
    print("="*60)
    
    # Line loss results
    line_results = pd.DataFrame({
        'Line': net.line.index,
        'From Bus': net.line.from_bus,
        'To Bus': net.line.to_bus,
        'P_loss (MW)': net.res_line.pl_mw,
        'Q_loss (MVAr)': net.res_line.ql_mvar,
        'Loading (%)': net.res_line.loading_percent
    })
    print(line_results.to_string(index=False))
    
    print("\n" + "="*60)
    print("BASE CASE RESULTS - SYSTEM POWER BALANCE")
    print("="*60)
    
    # Calculate power balance
    total_gen_p = net.res_ext_grid.p_mw.sum() if hasattr(net, 'res_ext_grid') and len(net.res_ext_grid) > 0 else 0
    total_gen_q = net.res_ext_grid.q_mvar.sum() if hasattr(net, 'res_ext_grid') and len(net.res_ext_grid) > 0 else 0
    
    # Add generator contributions if present
    if hasattr(net, 'gen') and len(net.gen) > 0:
        total_gen_p += net.res_gen.p_mw.sum()
        total_gen_q += net.res_gen.q_mvar.sum()
    
    # Add sgen (static generator) contributions if present
    if hasattr(net, 'sgen') and len(net.sgen) > 0:
        total_gen_p += net.res_sgen.p_mw.sum()
        total_gen_q += net.res_sgen.q_mvar.sum()
    
    total_load_p = net.load.p_mw.sum() if hasattr(net, 'load') and len(net.load) > 0 else 0
    total_load_q = net.load.q_mvar.sum() if hasattr(net, 'load') and len(net.load) > 0 else 0
    
    total_loss_p = net.res_line.pl_mw.sum()
    total_loss_q = net.res_line.ql_mvar.sum()
    
    # Add transformer losses if present
    if hasattr(net, 'trafo') and len(net.trafo) > 0:
        total_loss_p += net.res_trafo.pl_mw.sum()
        total_loss_q += net.res_trafo.ql_mvar.sum()
    
    balance_p = total_gen_p - total_load_p - total_loss_p
    balance_q = total_gen_q - total_load_q - total_loss_q
    
    power_balance = {
        'Component': ['Total Generation (MW)', 'Total Generation (MVAr)',
                      'Total Load (MW)', 'Total Load (MVAr)',
                      'Total Losses (MW)', 'Total Losses (MVAr)',
                      'Power Balance Error (MW)', 'Power Balance Error (MVAr)'],
        'Value': [f"{total_gen_p:.4f}", f"{total_gen_q:.4f}",
                  f"{total_load_p:.4f}", f"{total_load_q:.4f}",
                  f"{total_loss_p:.4f}", f"{total_loss_q:.4f}",
                  f"{balance_p:.6f}", f"{balance_q:.6f}"]
    }
    
    df_power_balance = pd.DataFrame(power_balance)
    print(df_power_balance.to_string(index=False))
    
    # --- 6. Export Results to CSV ---
    if EXPORT_RESULTS:
        print(f"\nExporting results to '{output_dir}' folder...")
        df_comparison.to_csv(f"{output_dir}/comparison.csv", index=False)
        if nr_success and fd_success:
            df_accuracy.to_csv(f"{output_dir}/accuracy_comparison.csv", index=False)
        bus_results.to_csv(f"{output_dir}/bus_voltages.csv", index=False)
        line_results.to_csv(f"{output_dir}/line_losses.csv", index=False)
        df_power_balance.to_csv(f"{output_dir}/power_balance.csv", index=False)
        print("✓ Results exported successfully!")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)

# --- 7. NETWORK TOPOLOGY VISUALIZATION ---
print("\n" + "="*60)
print("GENERATING NETWORK TOPOLOGY DIAGRAM")
print("="*60)

try:
    import matplotlib.pyplot as plt
    import pandapower.plotting as plot
    
    # Re-run NR to ensure results are loaded
    pp.runpp(net, algorithm='nr')
    
    # Create network topology plot
    print("Creating network topology diagram...")
    
    # Simple network plot
    plt.figure(figsize=(16, 12))
    
    try:
        # Try to use the built-in plotting if geocoordinates exist
        plot.simple_plot(net, 
                        plot_loads=True, 
                        plot_gens=True, 
                        plot_line_switches=True,
                        line_color='gray',
                        bus_size=1.5,
                        line_width=2.0,
                        load_size=1.0,
                        gen_size=1.5,
                        ext_grid_size=2.0,
                        show_plot=False)
        plt.title(f'{TARGET_SYSTEM.upper()} - Distribution System Topology', fontsize=16, fontweight='bold', pad=20)
    except:
        # If simple_plot fails, create using create_generic_coordinates
        print("Generating generic coordinates for topology...")
        plot.create_generic_coordinates(net, respect_switches=True)
        plot.simple_plot(net, 
                        plot_loads=True, 
                        plot_gens=True,
                        line_color='gray',
                        bus_size=1.5,
                        line_width=2.0,
                        load_size=1.0,
                        gen_size=1.5,
                        ext_grid_size=2.0,
                        show_plot=False)
        plt.title(f'{TARGET_SYSTEM.upper()} - Distribution System Topology', fontsize=16, fontweight='bold', pad=20)
    
    if EXPORT_RESULTS:
        plt.savefig(f"{output_dir}/network_topology.png", dpi=300, bbox_inches='tight')
        print(f"✓ Network topology saved to '{output_dir}/network_topology.png'")
    
    plt.show()
    
except Exception as e:
    print(f"Warning: Could not generate network topology plot: {e}")

# --- 8. ANALYSIS RESULTS VISUALIZATION ---
print("\n" + "="*60)
print("GENERATING ANALYSIS PLOTS")
print("="*60)

import matplotlib.pyplot as plt

if nr_success:
    # Re-run NR to ensure results are loaded
    pp.runpp(net, algorithm='nr')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Load Flow Analysis Results - {TARGET_SYSTEM}', fontsize=16, fontweight='bold')
    
    # Plot 1: Voltage Profile
    axes[0, 0].plot(net.res_bus.vm_pu, 'b-o', linewidth=2, markersize=4)
    axes[0, 0].axhline(y=1.0, color='g', linestyle='--', label='Nominal (1.0 p.u.)')
    axes[0, 0].axhline(y=0.95, color='r', linestyle='--', label='Min Limit (0.95 p.u.)')
    axes[0, 0].set_xlabel('Bus Number')
    axes[0, 0].set_ylabel('Voltage Magnitude (p.u.)')
    axes[0, 0].set_title('Bus Voltage Profile')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Voltage Angle
    axes[0, 1].plot(net.res_bus.va_degree, 'r-o', linewidth=2, markersize=4)
    axes[0, 1].set_xlabel('Bus Number')
    axes[0, 1].set_ylabel('Voltage Angle (degrees)')
    axes[0, 1].set_title('Bus Voltage Angle')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Line Losses
    axes[1, 0].bar(net.res_line.index, net.res_line.pl_mw, color='orange', alpha=0.7)
    axes[1, 0].set_xlabel('Line Number')
    axes[1, 0].set_ylabel('Active Power Loss (MW)')
    axes[1, 0].set_title('Line Power Losses')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Line Loading
    axes[1, 1].bar(net.res_line.index, net.res_line.loading_percent, color='purple', alpha=0.7)
    axes[1, 1].axhline(y=100, color='r', linestyle='--', label='Max Loading (100%)')
    axes[1, 1].set_xlabel('Line Number')
    axes[1, 1].set_ylabel('Loading (%)')
    axes[1, 1].set_title('Line Loading Percentage')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save plot
    if EXPORT_RESULTS:
        plt.savefig(f"{output_dir}/analysis_plots.png", dpi=300, bbox_inches='tight')
        print(f"✓ Analysis plots saved to '{output_dir}/analysis_plots.png'")
    
    plt.show()

print("\n" + "="*60)
print("ALL VISUALIZATIONS COMPLETE")
print("="*60)
