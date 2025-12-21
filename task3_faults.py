import pandapower as pp
import pandapower.networks as nw
import pandapower.shortcircuit as sc
import numpy as np

# --- 1. SETUP ---
net = nw.case33bw()
fault_bus = 1  # We analyze Bus 2 (Index 1). Index 0 is the Source.

# DEFINE GRID PARAMETERS
vn_kv = 12.66        # Nominal Voltage
s_sc_mva = 1000.0    # Grid Short Circuit Capacity
c_factor = 1.1       # IEC 60909 Voltage Factor (Safety Margin)

# Apply to Network
net.ext_grid['s_sc_max_mva'] = s_sc_mva
net.ext_grid['rx_max'] = 0.1
net.ext_grid['x0x_max'] = 1.0  # Z0 = Z1 for Grid
net.ext_grid['r0x0_max'] = 0.1

# DEFINE LINE PARAMETERS (Approximation for Distribution)
# Z0 = 3 * Z1 for lines
net.line['r0_ohm_per_km'] = net.line['r_ohm_per_km'] * 3
net.line['x0_ohm_per_km'] = net.line['x_ohm_per_km'] * 3
net.line['c0_nf_per_km'] = net.line['c_nf_per_km']

print(f"--- VALIDATION ANALYSIS: Bus {fault_bus+1} (Index {fault_bus}) ---")

# --- 2. RUN SIMULATION (The "Black Box") ---
sc.calc_sc(net, fault='1ph', ip=True, ith=True)
i_sim = net.res_bus_sc.ikss_ka[fault_bus]
print(f"1. Simulation Result (Pandapower): {i_sim:.4f} kA")

# --- 3. MANUAL ANALYTICAL CALCULATION (The "White Box") ---
# We calculate the impedance from the Source (Bus 0) to the Fault (Bus 1)

# A. Calculate Source Impedance (Z_grid)
# Formula: Z_grid = (c * Vn^2) / S_sc
z_grid_mag = (c_factor * vn_kv**2) / s_sc_mva
# Split into R and X (assuming purely reactive for simplicity or small R)
r_grid = 0.1 * z_grid_mag / np.sqrt(1 + 0.1**2) # using R/X = 0.1
x_grid = z_grid_mag / np.sqrt(1 + 0.1**2)
Z_grid_1 = complex(r_grid, x_grid)
Z_grid_0 = Z_grid_1 # Because we set x0x_max = 1.0

# B. Calculate Line Impedance (Bus 0 -> Bus 1)
# We get the line connecting the source to the fault bus
line_idx = net.line[(net.line.from_bus == 0) & (net.line.to_bus == 1)].index[0]
r_line = net.line.at[line_idx, 'r_ohm_per_km'] * net.line.at[line_idx, 'length_km']
x_line = net.line.at[line_idx, 'x_ohm_per_km'] * net.line.at[line_idx, 'length_km']

Z_line_1 = complex(r_line, x_line)
Z_line_0 = complex(r_line * 3, x_line * 3) # Our assumption Z0 = 3*Z1

# C. Total Thevenin Impedance at Fault Location
Z_total_1 = Z_grid_1 + Z_line_1
Z_total_2 = Z_total_1 # Z2 = Z1 in static networks
Z_total_0 = Z_grid_0 + Z_line_0

# D. Calculate Fault Current
# Formula: I = (3 * c * V_ph) / (Z1 + Z2 + Z0)
# Note: c * Vn is Line-to-Line, so Phase voltage is (c * Vn / sqrt(3))
# The 3 cancels out sqrt(3) -> numerator becomes sqrt(3) * c * Vn
numerator = np.sqrt(3) * c_factor * vn_kv
denominator = Z_total_1 + Z_total_2 + Z_total_0
i_calc = numerator / abs(denominator)

print(f"2. Manual Calculation:")
print(f"   Z_pos (Grid+Line): {Z_total_1:.4f} Ohm")
print(f"   Z_zero (Grid+Line):{Z_total_0:.4f} Ohm")
print(f"   Z_Loop (|2Z1+Z0|): {abs(denominator):.4f} Ohm")
print(f"   Calculated Current:{i_calc:.4f} kA")

# --- 4. CHECK ---
diff = abs(i_sim - i_calc)
print(f"\nDifference: {diff:.4f} kA")
if diff < 0.1:
    print("VALIDATION SUCCESSFUL")
else:
    print("VALIDATION FAILED")