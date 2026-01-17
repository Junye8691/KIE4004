"""
KIE4004 – Task 1
Load Flow Analysis for IEEE Test Systems

Comparison between:
- Newton–Raphson
- Fast Decoupled Load Flow

Systems supported:
- IEEE 33-Bus
- IEEE 69-Bus
- IEEE 118-Bus

"""

import pandapower as pp
import pandapower.networks as nw
import time
import pandas as pd
from pandapower.converter.matpower import from_mpc
import os

# ============================================================
# ================= NETWORK BUILDER ==========================
# ============================================================

def build_network(system):
    """
    Load IEEE test systems:
    - 33-bus, 118-bus from pandapower
    - 69-bus from MATPOWER (.m file)
    """

    if system == "33":
        net = nw.case33bw()

    elif system == "118":
        net = nw.case118()

    elif system == "69":
        # === Load from MATPOWER ===
        base_path = os.path.dirname(os.path.abspath(__file__))
        m_file = os.path.join(base_path, "matpower", "data", "case69.m")

        if not os.path.exists(m_file):
            raise FileNotFoundError("case69.m not found in matpower/data/")

        net = from_mpc(m_file, f_hz=50)

        # === UNIT CORRECTION ===
        # MATPOWER uses kW / kVAr, pandapower uses MW / MVAr
        net.load.p_mw /= 1000.0
        net.load.q_mvar /= 1000.0

        # Fix line impedance (undo p.u. conversion)
        base_mva = 10.0
        base_kv = 12.66
        z_base = (base_kv ** 2) / base_mva

        net.line.r_ohm_per_km /= z_base
        net.line.x_ohm_per_km /= z_base

    else:
        raise ValueError("Unsupported system. Choose '33', '69', or '118'.")

    # Ensure slack bus exists
    if len(net.ext_grid) == 0:
        pp.create_ext_grid(net, bus=0, vm_pu=1.0)

    return net


# ============================================================
# ================= SOLVER RUNNER ============================
# ============================================================

def run_solver(net, algorithm, tol, max_iter):
    """
    Run power flow and return performance metrics
    """
    pp.reset_results(net)
    start_time = time.time()

    try:
        pp.runpp(
            net,
            algorithm=algorithm,
            tolerance_mva=tol,
            max_iteration=max_iter
        )

        exec_time = time.time() - start_time

        try:
            iterations = net._ppc["iterations"]
        except:
            iterations = "N/A"

        total_loss = net.res_line.pl_mw.sum()
        min_voltage = net.res_bus.vm_pu.min()

        return {
            "Status": "Converged",
            "Time (s)": round(exec_time, 5),
            "Iterations": iterations,
            "Total Loss (MW)": round(total_loss, 4),
            "Min Voltage (p.u.)": round(min_voltage, 4)
        }

    except pp.LoadflowNotConverged:
        return {
            "Status": "Did Not Converge",
            "Time (s)": "-",
            "Iterations": max_iter,
            "Total Loss (MW)": "-",
            "Min Voltage (p.u.)": "-"
        }


# ============================================================
# ================= MAIN INTERFACE ===========================
# ============================================================

def run_powerflow(system="33", method="ALL", tol=1e-6, max_iter=50):
    """
    Main Task 1 interface
    """
    print("\n")
    print(f"{system}-BUS LOAD FLOW ANALYSIS")
    print("==============================================")

    net = build_network(system)
    print(f"Number of buses: {len(net.bus)}")

    results = {}

    if method in ["NR", "ALL"]:
        print("\nRunning Newton–Raphson...")
        results["Newton–Raphson"] = run_solver(
            net, algorithm="nr", tol=tol, max_iter=max_iter
        )

    if method in ["FDLF", "ALL"]:
        print("\nRunning Fast Decoupled...")
        results["Fast Decoupled"] = run_solver(
            net, algorithm="fdbx", tol=tol, max_iter=max_iter
        )

    df = pd.DataFrame(results)

    print("\n")
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    print(df)
    print("\n")

    return df


# ============================================================
# ================= STANDALONE RUN ===========================
# ============================================================

if __name__ == "__main__":
    # Change parameters here if running directly
    run_powerflow(
        system="69",     # "33", "69", or "118"
        method="ALL",    # "NR", "FDLF", or "ALL"
        tol=1e-6,
        max_iter=50
    )
