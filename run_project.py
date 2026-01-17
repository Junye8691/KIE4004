"""
KIE4004 – Power System Project

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

# Select tasks (can run multiple)
TASKS = ["loadflow", "renewable", "fault"]  
# Options: "loadflow", "renewable", "fault"

# Load flow method (only used if TASK == "loadflow")
METHOD = "ALL"        # Options: "NR", "FDLF", "ALL"

# Numerical settings
TOL = 1e-6
MAX_ITER = 50

# Fault settings (only used if TASK == "fault")
FAULT_BUS = 18
FAULT_TYPE = "LG"    # Options: "LG", "LL", "LLG"

# Renewable settings (only used if TASK == "renewable")
RE_BUS = 18
MAX_RE_MW = 1.0

# ============================================================
# ===================== IMPORT SECTION =======================
# ============================================================

import sys
import os

# Ensure local MATPOWER folder is used
sys.path.append(os.path.join(os.getcwd(), "matpower"))

# Import task modules
import task1_analysis as pf
import task2_renewable as re
import task3_faults as fault

# ============================================================
# ======================= MAIN LOGIC =========================
# ============================================================

def main():
    print("\n")
    print(" KIE4004 POWER SYSTEM PROJECT ")
    print("==============================================")

    print(f"System        : IEEE {SYSTEM}-Bus")
    print(f"Selected Task : {TASKS}")

    for task in TASKS:

        if task == "loadflow":
            print("\n>>> Running Load Flow Analysis")
            run_loadflow()

        elif task == "renewable":
            print("\n>>> Running Renewable Integration")
            run_renewable()

        elif task == "fault":
            print("\n>>> Running Fault Analysis")
            run_fault()

        else:
            raise ValueError(f"Invalid task: {task}")

    print("\nSimulation completed successfully")


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
        max_re_mw=MAX_RE_MW
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
