"""
KIE4004 – Power System Project
Master Runner Script (Live Demo Friendly)

Author: Group KIE4004
Role: Member 1 (Architect)

USAGE:
1. Edit PARAMETERS section only
2. Run this file
3. Results will be printed and plotted automatically
"""

# ============================================================
# ======================= PARAMETERS =========================
# ============================================================

# Select system size
SYSTEM = "33"        # Options: "33", "69", "118"

# Select task
TASK = "loadflow"    # Options: "loadflow", "renewable", "fault"

# Load flow method (only used if TASK == "loadflow")
METHOD = "NR"        # Options: "NR", "FDLF"

# Numerical settings
TOL = 1e-6
MAX_ITER = 50

# Fault settings (only used if TASK == "fault")
FAULT_BUS = 18
FAULT_TYPE = "LG"    # Options: "LG", "LL", "LLG"

# Renewable settings (only used if TASK == "renewable")
RE_BUS = 18
RE_P_MW = 1.0

# ============================================================
# ===================== IMPORT SECTION =======================
# ============================================================

import sys
import os

# Ensure local MATPOWER folder is used
sys.path.append(os.path.join(os.getcwd(), "matpower"))

# Import task modules
import task_1_analysis as pf
import task_2_renewable as re
import task_3_faults as fault

# ============================================================
# ======================= MAIN LOGIC =========================
# ============================================================

def main():
    print("\n==============================================")
    print(" KIE4004 POWER SYSTEM PROJECT – MASTER RUNNER ")
    print("==============================================")

    print(f"System        : IEEE {SYSTEM}-Bus")
    print(f"Selected Task : {TASK}")

    if TASK == "loadflow":
        print(f"Method        : {METHOD}")
        print(f"Tolerance     : {TOL}")
        print(f"Max Iter      : {MAX_ITER}")
        run_loadflow()

    elif TASK == "renewable":
        print(f"RE Bus        : {RE_BUS}")
        print(f"RE Size (MW)  : {RE_P_MW}")
        run_renewable()

    elif TASK == "fault":
        print(f"Fault Bus     : {FAULT_BUS}")
        print(f"Fault Type   : {FAULT_TYPE}")
        run_fault()

    else:
        raise ValueError("Invalid TASK selected")

    print("\n✅ Simulation completed successfully")
    print("==============================================\n")


# ============================================================
# ==================== TASK FUNCTIONS ========================
# ============================================================

def run_loadflow():
    """
    Task 1 – Newton-Raphson vs Fast Decoupled
    """
    pf.run_powerflow(
        system=SYSTEM,
        method=METHOD,
        tol=TOL,
        max_iter=MAX_ITER
    )


def run_renewable():
    """
    Task 2 – Renewable Energy Integration
    """
    re.run_renewable_analysis(
        system=SYSTEM,
        re_bus=RE_BUS,
        re_p_mw=RE_P_MW
    )


def run_fault():
    """
    Task 3 – Fault Analysis
    """
    fault.run_fault_analysis(
        system=SYSTEM,
        fault_bus=FAULT_BUS,
        fault_type=FAULT_TYPE
    )


# ============================================================
# ====================== ENTRY POINT =========================
# ============================================================

if __name__ == "__main__":
    main()
