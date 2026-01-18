import pandapower as pp
import pandapower.networks as nw
from pandapower.converter.matpower import from_mpc
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def save_figure(fig_name, task_folder):
    base_path = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_path, task_folder)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, fig_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[Saved] {save_path}")

# =========================================

def get_network(sys_name, verbose=False):
    if verbose:
        print(f"Loading system: {sys_name}...")

    if sys_name == "case33bw":
        return nw.case33bw()

    if sys_name == "case118":
        return nw.case118()

    base_path = os.path.dirname(os.path.abspath(__file__))
    m_file_path = os.path.join(base_path, 'matpower', 'data', f'{sys_name}.m')

    net = from_mpc(m_file_path, f_hz=50)

    if sys_name == 'case69':
        if verbose:
            print("Applying unit correction for IEEE 69-bus...")

        net.load.p_mw /= 1000.0
        net.load.q_mvar /= 1000.0

        base_mva = 10.0
        base_kv = 12.66
        z_base = (base_kv ** 2) / base_mva

        net.line.r_ohm_per_km /= z_base
        net.line.x_ohm_per_km /= z_base

    return net

def calculate_vsi(net):
    """
    Calculates the Voltage Stability Index (VSI) for each bus.
    VSI = |V_send|^4 - 4 * (P * X - Q * R)^2 - 4 * |V_send|^2 * (P * R + Q * X)
    For radial systems, a simplified check is often used on branches.
    Low VSI (near 0) = Unstable. High VSI (near 1) = Stable.
    """
    # EASIER ALTERNATIVE FOR PRESENTATION:
    # Just calculate the "Voltage Deviation Index" (VDI)
    # VDI = Sum((V_nominal - V_actual)^2)
    v_nominal = 1.0
    vdi = sum((v_nominal - net.res_bus.vm_pu)**2)
    return vdi

#case118
#def run_renewable_analysis(system="118", re_bus=75, max_re_mw=500, step=25):

#case33
def run_renewable_analysis(system="33", re_bus=17, max_re_mw=3.5, step=0.2):
    
#case69
#def run_renewable_analysis(system="69", re_bus=64, max_re_mw=3.0, step=0.2):


    # ================= CONFIG =================
    if system == "33":
        TARGET_SYSTEM = "case33bw"
    elif system == "69":
        TARGET_SYSTEM = "case69"
    elif system == "118":
        TARGET_SYSTEM = "case118"
    else:
        raise ValueError("Unsupported system")
    WEAKEST_BUS = re_bus
    MAX_RE_MW = max_re_mw
    STEP = step

    # ============ BASE CASE ============
    net_base = get_network(TARGET_SYSTEM, verbose=True)

    if len(net_base.ext_grid) == 0:
        pp.create_ext_grid(net_base, bus=0, vm_pu=1.0)

    pp.runpp(net_base, numba=False)


    base_vm = net_base.res_bus.vm_pu.copy()
    base_min_v = base_vm.min()
    base_min_bus = base_vm.idxmin()
    base_loss = net_base.res_line.pl_mw.sum()
    base_vdi = calculate_vsi(net_base)
    

    print("\n--- BASE CASE ---")
    print(f"Minimum Voltage : {base_min_v:.4f} p.u. at Bus {base_min_bus}")
    print(f"Base Case Total Loss : {base_loss:.4f} MW")
    print(f"Stability (VDI) : {base_vdi:.4f} (Lower is better)")


    # ============ RE SIZE SWEEP ============
    sizes = np.arange(0, MAX_RE_MW + STEP, STEP)
    losses = []
    min_voltages = []
    vdis = []

    print(f"\n--- Optimizing RE Size at Bus {WEAKEST_BUS} ({TARGET_SYSTEM}) ---")
    print(f"{'Size (MW)':<10} | {'Loss (MW)':<10} | {'Min V (p.u.)':<12} | {'Stability (VDI)':<15} | {'Rev Flow?'}")

    

    for size in sizes:
        net = get_network(TARGET_SYSTEM, verbose=False)

        if len(net.ext_grid) == 0:
            pp.create_ext_grid(net, bus=0, vm_pu=1.0)

        pp.create_sgen(net, bus=WEAKEST_BUS, p_mw=size, q_mvar=0)

        pp.runpp(net, numba=False)

        loss = net.res_line.pl_mw.sum()
        v_min = net.res_bus.vm_pu.min()
        v_min_bus = net.res_bus.vm_pu.idxmin()
        vdi = calculate_vsi(net)

        losses.append(loss)
        min_voltages.append(v_min)
        vdis.append(vdi)

        
        
        #Reverse Power Flow Calculation
        p_slack = net.res_ext_grid.p_mw.sum()
        reverse_flow = "YES" if p_slack < 0 else "No"
        print(f"{size:<10.1f} | {loss:<10.4f} | {v_min:<12.4f} | {vdi:<15.4f} | {reverse_flow}")

    # ============ OPTIMAL SIZE ============
    optimal_idx = np.argmin(losses)
    optimal_size = sizes[optimal_idx]
    optimal_loss = losses[optimal_idx]
    optimal_vdi = vdis[optimal_idx]

    print("\n======================================")
    print(f"OPTIMAL RE SIZE : {optimal_size:.1f} MW")
    print(f"MINIMUM LOSS   : {optimal_loss:.4f} MW")
    print(f"STABILITY IMPRV : {base_vdi:.4f} -> {optimal_vdi:.4f} (VDI)")
    print("======================================")


    # ============ LOSS VS RE SIZE PLOT ============
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, losses, marker='o', label='Total System Loss')
    plt.axvline(optimal_size, linestyle='--',
                label=f'Optimal Size = {optimal_size:.1f} MW')
    plt.xlabel('RE Capacity (MW)')
    plt.ylabel('Total Active Power Loss (MW)')
    plt.title(f'RE Size Optimization at Bus {WEAKEST_BUS} ({TARGET_SYSTEM})')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_figure(
    fig_name=f"RE Size Optimization at Bus {WEAKEST_BUS} ({TARGET_SYSTEM}).png",
    task_folder="task2_results"
    )
    plt.show(block=False)


    # ============ VOLTAGE PROFILE AT OPTIMAL RE ============
    net_opt = get_network(TARGET_SYSTEM, verbose=False)

    if len(net_opt.ext_grid) == 0:
        pp.create_ext_grid(net_opt, bus=0, vm_pu=1.0)

    pp.create_sgen(net_opt, bus=WEAKEST_BUS, p_mw=optimal_size, q_mvar=0)

    pp.runpp(net_opt, numba=False)

    opt_vm = net_opt.res_bus.vm_pu.copy()


    # ============ VOLTAGE PROFILE COMPARISON PLOT ============
    plt.figure(figsize=(10, 5))

    plt.plot(base_vm.index, base_vm.values,
            marker='o', label='Base Case')

    plt.plot(opt_vm.index, opt_vm.values,
            marker='s', label=f'With RE ({optimal_size:.1f} MW)')

    plt.axhline(0.95, linestyle='--', linewidth=1)
    plt.axhline(1.05, linestyle='--', linewidth=1)

    plt.xlabel('Bus Number')
    plt.ylabel('Voltage Magnitude (p.u.)')
    plt.title(f'Voltage Profile Comparison ({TARGET_SYSTEM})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_figure(
    fig_name=f"Voltage Profile Comparison ({TARGET_SYSTEM}).png",
    task_folder="task2_results"
    )
    plt.show(block=False)

    plt.show()

if __name__ == "__main__":
    run_renewable_analysis()
