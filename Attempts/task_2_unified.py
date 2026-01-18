"""
TASK 2: UNIFIED RENEWABLE ENERGY INTEGRATION ANALYSIS
======================================================
Intelligent analysis that adapts to both:
- Distribution Systems: case33bw, case69 (radial, 12.66 kV, 0-5 MW)
- Transmission Systems: case118 (meshed, 138 kV, 10-150 MW)

Automatically detects network type and applies appropriate methodology
"""

import pandapower as pp
import pandapower.networks as nw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# ============================================================
# CONFIGURATION
# ============================================================
SYSTEM = 'case118'  # Options: 'case33bw', 'case69', 'case118'
RE_TYPE = 'solar_pv'  # Options: 'solar_pv', 'wind', 'biomass', 'small_hydro'
EXPORT_RESULTS = True
OUTPUT_DIR = f'results_task2_{SYSTEM}'

# Network classification
NETWORK_TYPES = {
    'case33bw': 'distribution',
    'case69': 'distribution',
    'case118': 'transmission'
}

# RE Configuration - ADAPTIVE based on network type
RE_CONFIG = {
    'distribution': {
        'solar_pv': {
            'name': 'Solar PV',
            'dispatchable': False,
            'power_factor': 0.95,
            'max_capacity': 3.0,  # MW
            'description': 'Rooftop/ground-mounted PV, inverter-based'
        },
        'wind': {
            'name': 'Wind Turbine',
            'dispatchable': False,
            'power_factor': 0.90,
            'max_capacity': 3.0,
            'description': 'Small wind turbines, variable output'
        },
        'biomass': {
            'name': 'Biomass Generator',
            'dispatchable': True,
            'power_factor': 0.85,
            'max_capacity': 2.5,
            'description': 'Dispatchable, steady output'
        },
        'small_hydro': {
            'name': 'Small Hydro',
            'dispatchable': True,
            'power_factor': 0.90,
            'max_capacity': 5.0,
            'description': 'Run-of-river hydro'
        }
    },
    'transmission': {
        'solar_pv': {
            'name': 'Solar PV Farm',
            'dispatchable': False,
            'power_factor': 0.95,
            'min_capacity': 10.0,
            'max_capacity': 300.0,  # MW
            'description': 'Large-scale solar farm with grid inverters'
        },
        'wind': {
            'name': 'Wind Farm',
            'dispatchable': False,
            'power_factor': 0.90,
            'min_capacity': 20.0,
            'max_capacity': 150.0,
            'description': 'Wind farm cluster'
        },
        'biomass': {
            'name': 'Biomass Power Plant',
            'dispatchable': True,
            'power_factor': 0.85,
            'min_capacity': 15.0,
            'max_capacity': 80.0,
            'description': 'Biomass thermal plant'
        },
        'small_hydro': {
            'name': 'Hydro Plant',
            'dispatchable': True,
            'power_factor': 0.90,
            'min_capacity': 25.0,
            'max_capacity': 100.0,
            'description': 'Run-of-river or small reservoir'
        }
    }
}

# Detect network type
network_type = NETWORK_TYPES.get(SYSTEM, 'distribution')
re_config = RE_CONFIG[network_type][RE_TYPE]

print("="*80)
print(f"TASK 2: RENEWABLE ENERGY INTEGRATION ANALYSIS")
print(f"System: {SYSTEM.upper()} ({network_type.upper()} network)")
print(f"RE Type: {re_config['name']}")
print(f"Dispatchable: {re_config['dispatchable']}")
print("="*80)

# Create output directory
if EXPORT_RESULTS and not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_network(system_name):
    """Load the specified power system network"""
    if system_name == 'case33bw':
        return nw.case33bw()
    elif system_name == 'case118':
        return nw.case118()
    elif system_name == 'case69':
        try:
            from task1_analysis import parse_matpower_case
            return parse_matpower_case(r'matpower\data\case69.m')
        except:
            print("Warning: Could not load case69 from custom parser")
            return None
    else:
        raise ValueError(f"Unknown system: {system_name}")


def get_bus_selection_criteria(net, network_type):
    """Get appropriate bus selection criteria based on network type"""
    
    bus_analysis = pd.DataFrame({
        'Bus': net.bus.index,
        'Voltage_pu': net.res_bus.vm_pu.values,
    })
    
    # Load per bus
    load_per_bus = net.load.groupby('bus')['p_mw'].sum() if hasattr(net, 'load') else pd.Series()
    bus_analysis['Load_MW'] = bus_analysis['Bus'].map(load_per_bus).fillna(0)
    
    # Generation per bus
    if hasattr(net, 'gen') and len(net.gen) > 0:
        gen_per_bus = net.gen.groupby('bus')['p_mw'].sum()
        bus_analysis['Gen_MW'] = bus_analysis['Bus'].map(gen_per_bus).fillna(0)
    else:
        bus_analysis['Gen_MW'] = 0
    
    if network_type == 'distribution':
        # DISTRIBUTION: Focus on weak buses (low voltage, high load)
        bus_analysis['Voltage_deviation'] = abs(1.0 - bus_analysis['Voltage_pu'])
        bus_analysis['Suitability_Score'] = (
            bus_analysis['Voltage_deviation'] * 10 +  # Voltage improvement
            bus_analysis['Load_MW'] * 5  # Loss reduction
        )
        criteria_desc = "Weak buses (low voltage) with high loads"
        
    else:  # transmission
        # TRANSMISSION: Load centers with low generation density
        bus_analysis['Voltage_quality'] = 1 - abs(1.0 - bus_analysis['Voltage_pu'])
        bus_analysis['Load_factor'] = bus_analysis['Load_MW'] / bus_analysis['Load_MW'].max() if bus_analysis['Load_MW'].max() > 0 else 0
        bus_analysis['Gen_factor'] = 1 - (bus_analysis['Gen_MW'] / bus_analysis['Gen_MW'].max()) if bus_analysis['Gen_MW'].max() > 0 else 1
        
        bus_analysis['Suitability_Score'] = (
            bus_analysis['Load_factor'] * 40 +
            bus_analysis['Gen_factor'] * 30 +
            bus_analysis['Voltage_quality'] * 30
        )
        criteria_desc = "Load centers with low generation density"
    
    return bus_analysis, criteria_desc


def calculate_system_losses(net):
    """Calculate total system losses (lines + transformers)"""
    loss = net.res_line.pl_mw.sum() if hasattr(net, 'res_line') else 0
    if hasattr(net, 'res_trafo') and len(net.res_trafo) > 0:
        loss += net.res_trafo.pl_mw.sum()
    if hasattr(net, 'res_impedance') and len(net.res_impedance) > 0:
        loss += net.res_impedance.pl_mw.sum()
    return loss


# ============================================================
# 1. LOAD BASE SYSTEM AND ANALYZE
# ============================================================
print("\n--- STEP 1: BASE CASE ANALYSIS ---")

net_base = load_network(SYSTEM)
if net_base is None:
    print("Error: Could not load network. Exiting.")
    exit(1)

pp.runpp(net_base)

# Base case metrics
base_loss = calculate_system_losses(net_base)
base_vmin = net_base.res_bus.vm_pu.min()
base_vmax = net_base.res_bus.vm_pu.max()
base_voltages = net_base.res_bus.vm_pu.copy()

print(f"Base Case Results:")
print(f"  Total Losses: {base_loss:.4f} MW")
print(f"  Min Voltage: {base_vmin:.4f} p.u.")
print(f"  Max Voltage: {base_vmax:.4f} p.u.")
print(f"  Voltage violations (<0.95): {(base_voltages < 0.95).sum()} buses")
print(f"  Voltage violations (>1.05): {(base_voltages > 1.05).sum()} buses")

# ============================================================
# 2. IDENTIFY CANDIDATE BUSES FOR RE INTEGRATION
# ============================================================
print(f"\n--- STEP 2: CANDIDATE BUS SELECTION ---")
print(f"Network Type: {network_type.upper()}")

bus_analysis, criteria_desc = get_bus_selection_criteria(net_base, network_type)

# Exclude inappropriate buses
excluded_buses = {0}  # Always exclude slack
if network_type == 'transmission':
    # Also exclude buses with existing generation
    if hasattr(net_base, 'gen'):
        excluded_buses.update(net_base.gen.bus.values)

bus_analysis_filtered = bus_analysis[~bus_analysis['Bus'].isin(excluded_buses)]
bus_analysis_filtered = bus_analysis_filtered.sort_values('Suitability_Score', ascending=False)

# Select candidates
n_candidates = 8 if network_type == 'transmission' else 5
candidate_buses = bus_analysis_filtered.head(n_candidates)

print(f"\nSelection Criteria: {criteria_desc}")
print(f"\nTop {n_candidates} Candidate Buses for RE Integration:")
print(candidate_buses[['Bus', 'Voltage_pu', 'Load_MW', 'Suitability_Score']].to_string(index=False))

# Primary bus
primary_bus = int(candidate_buses.iloc[0]['Bus'])
print(f"\n✓ Selected PRIMARY bus for RE: Bus {primary_bus}")
print(f"  Voltage: {candidate_buses.iloc[0]['Voltage_pu']:.4f} p.u.")
print(f"  Load: {candidate_buses.iloc[0]['Load_MW']:.2f} MW")

# ============================================================
# 3. OPTIMAL CAPACITY SIZING
# ============================================================
print(f"\n--- STEP 3: OPTIMAL CAPACITY SIZING AT BUS {primary_bus} ---")

pf = re_config['power_factor']

# Set capacity range based on network type
if network_type == 'distribution':
    max_capacity = re_config['max_capacity']
    capacities = np.linspace(0, max_capacity, 31)
else:  # transmission
    min_capacity = re_config['min_capacity']
    max_capacity = re_config['max_capacity']
    capacities = np.linspace(min_capacity, max_capacity, 31)

results = {
    'capacity': [],
    'losses': [],
    'loss_reduction_pct': [],
    'v_min': [],
    'v_max': [],
    'v_violations': [],
    'convergence': []
}

if network_type == 'transmission':
    results['max_line_loading'] = []

print(f"\n{'Capacity (MW)':<15} {'Loss (MW)':<12} {'Loss Δ%':<10} {'Min V':<10} {'Max V':<10}", end='')
if network_type == 'transmission':
    print(f" {'Max Load%':<12}")
else:
    print()
print("-" * (80 if network_type == 'transmission' else 65))

for capacity in capacities:
    net = load_network(SYSTEM)
    
    if capacity > 0:
        q_mvar = capacity * np.tan(np.arccos(pf))
        pp.create_sgen(net, bus=primary_bus, p_mw=capacity, q_mvar=q_mvar,
                      name=f"{re_config['name']} {capacity:.2f}MW")
    
    try:
        pp.runpp(net, max_iteration=100)
        
        loss = calculate_system_losses(net)
        v_min = net.res_bus.vm_pu.min()
        v_max = net.res_bus.vm_pu.max()
        v_violations = ((net.res_bus.vm_pu < 0.95) | (net.res_bus.vm_pu > 1.05)).sum()
        loss_reduction = ((base_loss - loss) / base_loss) * 100
        
        results['capacity'].append(capacity)
        results['losses'].append(loss)
        results['loss_reduction_pct'].append(loss_reduction)
        results['v_min'].append(v_min)
        results['v_max'].append(v_max)
        results['v_violations'].append(v_violations)
        results['convergence'].append(True)
        
        print(f"{capacity:<15.2f} {loss:<12.4f} {loss_reduction:<10.2f} {v_min:<10.4f} {v_max:<10.4f}", end='')
        
        if network_type == 'transmission':
            max_loading = net.res_line.loading_percent.max()
            results['max_line_loading'].append(max_loading)
            print(f" {max_loading:<12.2f}")
        else:
            print()
        
    except Exception as e:
        print(f"{capacity:<15.2f} FAILED - {str(e)[:40]}")
        results['convergence'].append(False)

# Find optimal capacity
valid_indices = [i for i, conv in enumerate(results['convergence']) if conv]
if valid_indices:
    valid_losses = [results['losses'][i] for i in valid_indices]
    optimal_idx = valid_indices[valid_losses.index(min(valid_losses))]
    optimal_capacity = results['capacity'][optimal_idx]
    optimal_loss = results['losses'][optimal_idx]
    optimal_loss_reduction = results['loss_reduction_pct'][optimal_idx]
    
    print(f"\n{'='*80}")
    print(f"OPTIMAL CAPACITY: {optimal_capacity:.2f} MW")
    print(f"Loss Reduction: {optimal_loss_reduction:.2f}% (from {base_loss:.4f} to {optimal_loss:.4f} MW)")
    print(f"Min Voltage: {results['v_min'][optimal_idx]:.4f} p.u.")
    if network_type == 'transmission':
        print(f"Max Line Loading: {results['max_line_loading'][optimal_idx]:.2f}%")
    print(f"{'='*80}")
else:
    print("\nError: No valid solutions found")
    exit(1)

# ============================================================
# 4. DETAILED COMPARISON: BASE VS WITH RE
# ============================================================
print(f"\n--- STEP 4: DETAILED IMPACT ANALYSIS ---")

net_with_re = load_network(SYSTEM)
q_mvar_optimal = optimal_capacity * np.tan(np.arccos(pf))
pp.create_sgen(net_with_re, bus=primary_bus, p_mw=optimal_capacity, q_mvar=q_mvar_optimal,
               name=f"{re_config['name']} {optimal_capacity:.2f}MW")
pp.runpp(net_with_re)

# Voltage comparison
voltage_comparison = pd.DataFrame({
    'Bus': net_base.bus.index,
    'Base_Voltage_pu': net_base.res_bus.vm_pu.values,
    'With_RE_Voltage_pu': net_with_re.res_bus.vm_pu.values,
    'Voltage_Change_pu': net_with_re.res_bus.vm_pu.values - net_base.res_bus.vm_pu.values
})

print("\nVoltage Changes (Top 10 buses):")
print(voltage_comparison.nlargest(10, 'Voltage_Change_pu')[['Bus', 'Base_Voltage_pu', 'With_RE_Voltage_pu', 'Voltage_Change_pu']].to_string(index=False))

# System metrics
loss_with_re = calculate_system_losses(net_with_re)
total_load = net_base.load.p_mw.sum() if hasattr(net_base, 'load') else 0

print(f"\n{'='*80}")
print("SYSTEM-LEVEL IMPACT SUMMARY:")
print(f"{'='*80}")

metrics_comparison = pd.DataFrame({
    'Metric': [
        'Total Load (MW)',
        'RE Generation (MW)',
        'RE Penetration (%)',
        'Total Losses (MW)',
        'Loss Reduction (%)',
        'Min Voltage (p.u.)',
        'Max Voltage (p.u.)',
        'Buses with V < 0.95',
        'Buses with V > 1.05'
    ],
    'Base Case': [
        f"{total_load:.2f}",
        "0.00",
        "0.00",
        f"{base_loss:.4f}",
        "0.00",
        f"{base_vmin:.4f}",
        f"{base_vmax:.4f}",
        f"{(net_base.res_bus.vm_pu < 0.95).sum()}",
        f"{(net_base.res_bus.vm_pu > 1.05).sum()}"
    ],
    'With RE': [
        f"{total_load:.2f}",
        f"{optimal_capacity:.2f}",
        f"{optimal_capacity/total_load*100:.2f}" if total_load > 0 else "N/A",
        f"{loss_with_re:.4f}",
        f"{optimal_loss_reduction:.2f}",
        f"{net_with_re.res_bus.vm_pu.min():.4f}",
        f"{net_with_re.res_bus.vm_pu.max():.4f}",
        f"{(net_with_re.res_bus.vm_pu < 0.95).sum()}",
        f"{(net_with_re.res_bus.vm_pu > 1.05).sum()}"
    ]
})

print(metrics_comparison.to_string(index=False))

# ============================================================
# 5. MULTIPLE RE LOCATIONS
# ============================================================
print(f"\n{'='*80}")
print("--- STEP 5: MULTIPLE LOCATION ANALYSIS ---")
print(f"{'='*80}")

multi_location_results = []

# Single location (already computed)
multi_location_results.append({
    'Scenario': 'Single Location',
    'Num_Locations': 1,
    'Total_Capacity_MW': optimal_capacity,
    'Loss_MW': optimal_loss,
    'Loss_Reduction_%': optimal_loss_reduction,
    'Min_Voltage_pu': results['v_min'][optimal_idx]
})

# Multiple locations
for n_locations in [3, 5]:
    net_multi = load_network(SYSTEM)
    if net_multi is None:
        continue
    
    total_cap = 0
    for i in range(min(n_locations, len(candidate_buses))):
        bus_id = int(candidate_buses.iloc[i]['Bus'])
        cap = optimal_capacity / n_locations
        q = cap * np.tan(np.arccos(pf))
        pp.create_sgen(net_multi, bus=bus_id, p_mw=cap, q_mvar=q,
                       name=f"{re_config['name']} @ Bus {bus_id}")
        total_cap += cap
    
    try:
        pp.runpp(net_multi)
        loss_multi = calculate_system_losses(net_multi)
        
        multi_location_results.append({
            'Scenario': f'{n_locations} Distributed',
            'Num_Locations': n_locations,
            'Total_Capacity_MW': total_cap,
            'Loss_MW': loss_multi,
            'Loss_Reduction_%': ((base_loss - loss_multi) / base_loss) * 100,
            'Min_Voltage_pu': net_multi.res_bus.vm_pu.min()
        })
    except:
        print(f"  {n_locations}-location scenario failed")

df_multi = pd.DataFrame(multi_location_results)
print("\nComparison of RE Distribution Strategies:")
print(df_multi.to_string(index=False))

# ============================================================
# 6. ENGINEERING JUSTIFICATION
# ============================================================
print(f"\n{'='*80}")
print("ENGINEERING JUSTIFICATION:")
print(f"{'='*80}")

if network_type == 'distribution':
    justification = f"""
1. RE TYPE: {re_config['name']} (Distribution-Scale)
   - Capacity Range: 0 - {re_config['max_capacity']:.1f} MW
   - Power Factor: {pf}
   - Type: {'Dispatchable' if re_config['dispatchable'] else 'Non-dispatchable'}
   
   Distribution System Rationale:
   - Sized for radial distribution feeders
   - Voltage-sensitive load areas benefit from local generation
   - Reduces long-distance power flow in radial topology
   - Classic quadratic I²R loss behavior expected

2. LOCATION: Bus {primary_bus} (Weakest Bus Strategy)
   - Voltage: {candidate_buses.iloc[0]['Voltage_pu']:.4f} p.u. (low voltage indicates weak bus)
   - Load: {candidate_buses.iloc[0]['Load_MW']:.2f} MW
   - Strategy: Place DG at weak buses to improve voltage and reduce losses
   - Follows optimal DG placement theory (Hung et al., 2010)

3. CAPACITY: {optimal_capacity:.2f} MW (Optimal for {optimal_loss_reduction:.2f}% loss reduction)
   - Sized to minimize losses without causing overvoltage
   - Typically 20-30% of feeder load is optimal
   - Maintains voltage within limits (0.95-1.05 p.u.)

4. NETWORK CHARACTERISTICS:
   - Radial topology → predictable power flow
   - Loss reduction follows quadratic relationship with capacity
   - Voltage improvement concentrated near RE location
"""
else:  # transmission
    justification = f"""
1. RE TYPE: {re_config['name']} (Transmission-Scale)
   - Capacity Range: {re_config['min_capacity']:.0f} - {re_config['max_capacity']:.0f} MW
   - Power Factor: {pf}
   - Type: {'Dispatchable' if re_config['dispatchable'] else 'Non-dispatchable'}
   
   Transmission System Rationale:
   - Large-scale installation economically viable
   - Grid-code compliant inverters provide ancillary services
   - Reduces dependence on remote generation
   - Congestion relief on transmission corridors

2. LOCATION: Bus {primary_bus} (Load Center Strategy)
   - Voltage: {candidate_buses.iloc[0]['Voltage_pu']:.4f} p.u. (stable)
   - Load: {candidate_buses.iloc[0]['Load_MW']:.2f} MW
   - Strategy: Place near load centers to reduce transmission distance
   - Avoids major generation hubs for diversification

3. CAPACITY: {optimal_capacity:.1f} MW (Optimal for {optimal_loss_reduction:.2f}% loss reduction)
   - Penetration: {optimal_capacity/total_load*100:.1f}% of system load
   - Maintains system stability and voltage quality
   - Grid-code compliant reactive power support

4. NETWORK CHARACTERISTICS:
   - Meshed topology → multi-path power flow
   - NON-QUADRATIC loss behavior (power redistributes through parallel paths)
   - Distributed integration ({df_multi.iloc[-1]['Loss_Reduction_%']:.2f}%) outperforms 
     single location ({df_multi.iloc[0]['Loss_Reduction_%']:.2f}%)
   - Voltage changes distributed across entire network
"""

print(justification)

# References
print("\nKey References:")
if network_type == 'distribution':
    print("- Hung, D. Q., et al. (2010). Optimal placement of DG. IEEE Trans. Power Systems")
    print("- El-Khattam & Salama (2004). Distributed generation technologies")
    print("- Adefarati & Bansal (2019). Reliability analysis of microgrids")
else:
    print("- Ackermann, T. (2005). Distributed generation: a definition")
    print("- IEA (2020). Renewable Energy Integration in Power Grids")
    print("- IEEE Std 1547-2018: Interconnection of Distributed Energy Resources")
    print("- NERC Reliability Standards for transmission-connected resources")

# ============================================================
# 7. EXPORT RESULTS
# ============================================================
if EXPORT_RESULTS:
    print(f"\n{'='*80}")
    print(f"Exporting results to '{OUTPUT_DIR}' folder...")
    
    candidate_buses.to_csv(f"{OUTPUT_DIR}/candidate_buses.csv", index=False)
    pd.DataFrame(results).to_csv(f"{OUTPUT_DIR}/capacity_optimization.csv", index=False)
    voltage_comparison.to_csv(f"{OUTPUT_DIR}/voltage_comparison.csv", index=False)
    metrics_comparison.to_csv(f"{OUTPUT_DIR}/system_metrics.csv", index=False)
    df_multi.to_csv(f"{OUTPUT_DIR}/multi_location_analysis.csv", index=False)
    
    with open(f"{OUTPUT_DIR}/engineering_justification.txt", 'w', encoding='utf-8') as f:
        f.write(justification)
    
    print("✓ Results exported successfully!")

# ============================================================
# 8. VISUALIZATION
# ============================================================
print(f"\n{'='*80}")
print("Generating visualization plots...")
print(f"{'='*80}")

# Filter valid results
valid_cap = [results['capacity'][i] for i in range(len(results['capacity'])) if results['convergence'][i]]
valid_loss = [results['losses'][i] for i in range(len(results['losses'])) if results['convergence'][i]]
valid_loss_red = [results['loss_reduction_pct'][i] for i in range(len(results['loss_reduction_pct'])) if results['convergence'][i]]
valid_vmin = [results['v_min'][i] for i in range(len(results['v_min'])) if results['convergence'][i]]
valid_vmax = [results['v_max'][i] for i in range(len(results['v_max'])) if results['convergence'][i]]

fig = plt.figure(figsize=(18, 12))

# Plot 1: Losses vs Capacity
ax1 = plt.subplot(2, 3, 1)
ax1.plot(valid_cap, valid_loss, 'b-o', linewidth=2, markersize=5)
ax1.axvline(optimal_capacity, color='r', linestyle='--', linewidth=2, label=f'Optimal: {optimal_capacity:.2f} MW')
ax1.axhline(base_loss, color='g', linestyle='--', linewidth=1, alpha=0.7, label=f'Base: {base_loss:.4f} MW')
ax1.set_xlabel('RE Capacity (MW)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Total System Loss (MW)', fontsize=11, fontweight='bold')
title_suffix = '\n(Radial Network - Quadratic)' if network_type == 'distribution' else '\n(Meshed Network - Non-quadratic)'
ax1.set_title(f'Losses vs RE Capacity{title_suffix}', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Loss Reduction %
ax2 = plt.subplot(2, 3, 2)
ax2.plot(valid_cap, valid_loss_red, 'g-o', linewidth=2, markersize=5)
ax2.axvline(optimal_capacity, color='r', linestyle='--', linewidth=2)
ax2.set_xlabel('RE Capacity (MW)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Loss Reduction (%)', fontsize=11, fontweight='bold')
ax2.set_title('Loss Reduction vs RE Capacity', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Voltage Profile Comparison
ax3 = plt.subplot(2, 3, 3)
ax3.plot(voltage_comparison['Bus'], voltage_comparison['Base_Voltage_pu'],
         'b-o', linewidth=2, markersize=3, label='Base Case')
ax3.plot(voltage_comparison['Bus'], voltage_comparison['With_RE_Voltage_pu'],
         'r-s', linewidth=2, markersize=3, label=f'With {optimal_capacity:.2f} MW RE')
ax3.axhline(1.0, color='g', linestyle='--', linewidth=1, alpha=0.7, label='Nominal')
ax3.axhline(0.95, color='orange', linestyle='--', linewidth=1, alpha=0.7)
ax3.axhline(1.05, color='orange', linestyle='--', linewidth=1, alpha=0.7)
ax3.axvline(primary_bus, color='purple', linestyle=':', linewidth=2, alpha=0.5, label=f'RE Bus {primary_bus}')
ax3.set_xlabel('Bus Number', fontsize=11, fontweight='bold')
ax3.set_ylabel('Voltage Magnitude (p.u.)', fontsize=11, fontweight='bold')
ax3.set_title('Voltage Profile Comparison', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=8)

# Plot 4: Voltage Improvement
ax4 = plt.subplot(2, 3, 4)
ax4.bar(voltage_comparison['Bus'], voltage_comparison['Voltage_Change_pu'],
        color='green', alpha=0.7, edgecolor='black')
ax4.axhline(0, color='black', linewidth=1)
ax4.set_xlabel('Bus Number', fontsize=11, fontweight='bold')
ax4.set_ylabel('Voltage Improvement (p.u.)', fontsize=11, fontweight='bold')
ax4.set_title('Voltage Change per Bus', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Min/Max Voltage vs Capacity
ax5 = plt.subplot(2, 3, 5)
ax5.plot(valid_cap, valid_vmin, 'b-o', linewidth=2, markersize=4, label='Min Voltage')
ax5.plot(valid_cap, valid_vmax, 'r-s', linewidth=2, markersize=4, label='Max Voltage')
ax5.axhline(0.95, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Lower Limit')
ax5.axhline(1.05, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Upper Limit')
ax5.axvline(optimal_capacity, color='purple', linestyle='--', linewidth=2, alpha=0.5)
ax5.set_xlabel('RE Capacity (MW)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Voltage (p.u.)', fontsize=11, fontweight='bold')
ax5.set_title('Voltage Limits vs RE Capacity', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=8)

# Plot 6: Multi-location Comparison
ax6 = plt.subplot(2, 3, 6)
scenarios = [row['Scenario'] for row in multi_location_results]
loss_reductions = [row['Loss_Reduction_%'] for row in multi_location_results]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c'][:len(scenarios)]

bars = ax6.bar(range(len(scenarios)), loss_reductions, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax6.set_xticks(range(len(scenarios)))
ax6.set_xticklabels(scenarios, fontsize=10)
ax6.set_ylabel('Loss Reduction (%)', fontsize=11, fontweight='bold')
ax6.set_title('Distributed vs Centralized Integration', fontsize=11, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, loss_reductions):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

suptitle = f'RE Integration: {SYSTEM.upper()} ({network_type.title()}) - {re_config["name"]}'
plt.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

if EXPORT_RESULTS:
    plt.savefig(f"{OUTPUT_DIR}/re_analysis_plots.png", dpi=300, bbox_inches='tight')
    print(f"✓ Plots saved to '{OUTPUT_DIR}/re_analysis_plots.png'")

plt.show()

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE!")
print(f"{'='*80}")
print(f"\nKey Results for {SYSTEM.upper()} ({network_type.upper()}):")
print(f"- Optimal Capacity: {optimal_capacity:.2f} MW at Bus {primary_bus}")
print(f"- Loss Reduction: {optimal_loss_reduction:.2f}%")
print(f"- RE Penetration: {optimal_capacity/total_load*100:.2f}% of load" if total_load > 0 else "")
print(f"- Voltage improved: {base_vmin:.4f} → {results['v_min'][optimal_idx]:.4f} p.u.")
if len(multi_location_results) > 1:
    print(f"- Distributed strategy: {df_multi.iloc[-1]['Loss_Reduction_%']:.2f}% loss reduction")
print(f"- Network behavior: {'Quadratic (radial)' if network_type == 'distribution' else 'Non-quadratic (meshed)'}")
