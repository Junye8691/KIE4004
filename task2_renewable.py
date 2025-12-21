import pandapower as pp
import pandapower.networks as nw
import matplotlib.pyplot as plt
import numpy as np

# 1. Load System
net = nw.case33bw()
weakest_bus = 17 # Based on your previous run

# 2. Sweep RE Capacities from 0 to 3 MW
sizes = np.arange(0, 3.1, 0.2) # 0, 0.2, 0.4 ... 3.0
losses = []
voltages = []

print(f"--- Optimizing RE Size at Bus {weakest_bus} ---")
print(f"{'Size (MW)':<10} | {'Loss (MW)':<10} | {'Min Voltage':<10}")

for size in sizes:
    # Reset net to remove old generator
    net = nw.case33bw()
    
    # Add Generator
    pp.create_sgen(net, bus=weakest_bus, p_mw=size, q_mvar=0)
    
    # Run PF
    pp.runpp(net)
    
    # Store results
    loss = net.res_line.pl_mw.sum()
    v_min = net.res_bus.vm_pu.min()
    
    losses.append(loss)
    voltages.append(v_min)
    
    print(f"{size:<10.1f} | {loss:<10.4f} | {v_min:<10.4f}")

# 3. Find Optimal
min_loss = min(losses)
optimal_idx = losses.index(min_loss)
optimal_size = sizes[optimal_idx]

print(f"\nOPTIMAL SIZE: {optimal_size} MW (Min Loss: {min_loss:.4f} MW)")

# 4. Plot (Save as image for report)
plt.figure(figsize=(10, 5))
plt.plot(sizes, losses, marker='o', label='System Losses')
plt.axvline(optimal_size, color='r', linestyle='--', label=f'Optimal: {optimal_size} MW')
plt.title(f'Impact of RE Size at Bus {weakest_bus} on System Losses')
plt.xlabel('RE Capacity (MW)')
plt.ylabel('Total Active Loss (MW)')
plt.grid(True)
plt.legend()
plt.show()