import pandapower as pp
import pandapower.networks as nw
import time
import pandas as pd

# --- CONFIGURATION ---
# Choose your system: 'case33bw', 'case118', or 'case69'
TARGET_SYSTEM = 'case33bw' 

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
    fd_success = True
except Exception as e:
    fd_success = False
    print(f"FDLF Failed: {e}")

# --- 3. Comparison Table (Required for Report) ---
results = {
    'Metric': ['Time (s)', 'Iterations', 'Total Loss (MW)', 'Min Voltage (p.u.)'],
    'Newton-Raphson': [f"{nr_time:.5f}", nr_iter, f"{nr_loss:.4f}", f"{nr_vmin:.4f}"] if nr_success else ["Fail"]*4,
    'Fast Decoupled': [f"{fd_time:.5f}", fd_iter, f"{fd_loss:.4f}", f"{fd_vmin:.4f}"] if fd_success else ["Fail"]*4
}

df = pd.DataFrame(results)
print("\n" + "="*40)
print(df.to_string(index=False))
print("="*40)

# --- 4. Plotting (Optional for visualization) ---
# Un-comment below if you have matplotlib installed
# import matplotlib.pyplot as plt
# plt.plot(net.res_bus.vm_pu, label='Voltage Profile')
# plt.legend()
# plt.show()