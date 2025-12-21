import pandapower as pp
import pandapower.networks as nw
import pandas as pd

# 1. Load the Base Case (IEEE 33)
net = nw.case33bw()

# 2. Run Base Case Simulation
pp.runpp(net)
base_loss = net.res_line.pl_mw.sum()
base_min_vm = net.res_bus.vm_pu.min()
weakest_bus = net.res_bus.vm_pu.idxmin() # Finds the bus ID with lowest voltage

print(f"--- BASE CASE ---")
print(f"Weakest Bus: {weakest_bus}")
print(f"Min Voltage: {base_min_vm:.4f} p.u.")
print(f"Total Loss:  {base_loss:.4f} MW")

# 3. Integrate Renewable Energy (Solar PV)
# We add a Static Generator (sgen) at the weakest bus
# P_mw = 2.0 MW (Active Power), Q_mvar = 0 (Unity Power Factor)
re_capacity = 2.0 
pp.create_sgen(net, bus=weakest_bus, p_mw=re_capacity, q_mvar=0, name="Solar PV")

# 4. Run Simulation with RE
pp.runpp(net)
new_loss = net.res_line.pl_mw.sum()
new_min_vm = net.res_bus.vm_pu.min()

print(f"\n--- WITH {re_capacity} MW SOLAR PV AT BUS {weakest_bus} ---")
print(f"Min Voltage: {new_min_vm:.4f} p.u. (Improved from {base_min_vm:.4f})")
print(f"Total Loss:  {new_loss:.4f} MW (Reduced from {base_loss:.4f})")

# 5. Check Improvement
loss_reduction = ((base_loss - new_loss) / base_loss) * 100
print(f"\nSystem Loss Reduced by: {loss_reduction:.2f}%")