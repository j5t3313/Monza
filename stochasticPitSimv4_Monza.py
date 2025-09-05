import fastf1
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import trange

import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

# Import practice session tire modeling
from fp1_fp2_tire_model_monza import (
    build_compound_models_from_practice, 
    get_tire_performance_from_practice_models
)

# Import extracted targeted parameters
try:
    from monza_targeted_parameters import *
    PARAMETERS_LOADED = True
    print("Using extracted targeted parameters from monza_targeted_parameters.py")
    
    # Package extracted parameters for passing to tire model
    EXTRACTED_PARAMS = {
        'TIRE_PERFORMANCE': TIRE_PERFORMANCE,
        'POSITION_PENALTIES': POSITION_PENALTIES,
        'DRIVER_ERROR_RATES': DRIVER_ERROR_RATES,
        'DRS_EFFECTIVENESS': DRS_EFFECTIVENESS
    }
except ImportError:
    print("Warning: monza_targeted_parameters.py not found. Run F1_Parameter_Extractor.py first.")
    print("Using fallback default parameters.")
    PARAMETERS_LOADED = False
    EXTRACTED_PARAMS = None

# Fallback parameters if extraction file not available
if not PARAMETERS_LOADED:
    POSITION_PENALTIES = {i: {'penalty': 0.05 * (i-1)} for i in range(1, 21)}  # Lower for Monza
    TIRE_PERFORMANCE = {
        'SOFT': {'offset': 0.0, 'degradation_rate': 0.12},
        'MEDIUM': {'offset': 0.37, 'degradation_rate': 0.07},
        'HARD': {'offset': 0.70, 'degradation_rate': 0.04}
    }
    DRIVER_ERROR_RATES = {
        'dry': {'base_error_rate': 0.008}
    }
    DRS_EFFECTIVENESS = {
        'median_advantage': 0.48,
        'usage_probability': 0.45
    }
    
    EXTRACTED_PARAMS = {
        'TIRE_PERFORMANCE': TIRE_PERFORMANCE,
        'POSITION_PENALTIES': POSITION_PENALTIES,
        'DRIVER_ERROR_RATES': DRIVER_ERROR_RATES,
        'DRS_EFFECTIVENESS': DRS_EFFECTIVENESS
    }

# Monza-specific parameters
MONZA_PARAMS = {
    'base_pace': 81.5,      # Monza lap time 
    'num_laps': 53,         
    'rain_probability': 0.0,   
    'sc_probability': 0.50,    # F1 Official
    'vsc_probability': 0.38,   # F1 Official
    'pit_time_loss': 23.71,     # F1 Official
    'fuel_consumption_rate': 110 / 53,
    'fuel_effect_per_kg': 0.035
}

# Build tire models from practice sessions (FP1 + FP2)
print("Building tire degradation models from practice sessions for Monza...")
compound_models, model_info = build_compound_models_from_practice(
    year=2025, 
    gp_name='Italian Grand Prix', 
    base_pace=MONZA_PARAMS['base_pace']
)

# Load 2025 qualifying results for grid positions
try:
    quali = fastf1.get_session(2025, 'Italian Grand Prix', 'Q')
    quali.load()
    quali_results = quali.results[['Abbreviation', 'Position']].copy()
    quali_results = quali_results.dropna()
    quali_results['GridPosition'] = quali_results['Position'].astype(int)
    print("Grid positions from 2025 Italian GP qualifying:")
    print(quali_results[['Abbreviation', 'GridPosition']].head(10))
except:
    print("2025 Italian GP qualifying data not available - using example grid positions")
    quali_results = None
    # Safety car simulation (adapted from Zandvoort)
def generate_sc_laps(num_laps, weather_conditions):
    """Generate safety car periods based on track and weather"""
    sc_laps = set()
    
    base_sc_prob = MONZA_PARAMS['sc_probability']
    
    if np.random.rand() < base_sc_prob:
        # Always dry conditions for Monza
        sc_start = np.random.choice(range(8, num_laps-5))
        sc_duration = np.random.choice(range(2, 5))
        
        sc_laps.update(range(sc_start, min(sc_start + sc_duration, num_laps)))
    
    return sc_laps

# Position penalty function (adapted from Zandvoort with lower penalties)
def get_position_penalty(grid_pos, current_lap, total_laps):
    """Calculate time penalty based on starting grid position and traffic"""
    
    if PARAMETERS_LOADED:
        base_penalty = get_position_penalty_extracted(grid_pos)
    else:
        # Lower penalties for Monza
        penalty_map = {
            1: 0.0, 2: 0.05, 3: 0.08, 4: 0.10, 5: 0.12,
            6: 0.15, 7: 0.18, 8: 0.20, 9: 0.22, 10: 0.25
        }
        base_penalty = penalty_map.get(min(grid_pos, 10), 0.25 + (grid_pos - 10) * 0.02)
    
    # Traffic spread
    traffic_factor = max(0.2, 1.0 - (current_lap / total_laps) * 0.8)
    
    return base_penalty * traffic_factor

# Race sim
def simulate_race(strategy, grid_position, compound_models, num_laps=None,
                 base_pace=None, rain_probability=None, car_performance_factor=1.0):
    """Simulate a complete race with practice-based tire models"""
    
    if num_laps is None:
        num_laps = MONZA_PARAMS['num_laps']
    if base_pace is None:
        base_pace = MONZA_PARAMS['base_pace']
    if rain_probability is None:
        rain_probability = MONZA_PARAMS['rain_probability']
    
    race_time = 0
    current_lap = 1
    pit_time_loss = MONZA_PARAMS['pit_time_loss']
    current_position = grid_position
    fuel_load = 110
    fuel_consumption_rate = MONZA_PARAMS['fuel_consumption_rate']
    
    # Generate race conditions - always dry for Monza
    weather_conditions = ['dry'] * num_laps
    sc_laps = generate_sc_laps(num_laps, weather_conditions)
    
    track_evolution_rate = -0.0025
    
    # VSC simulation
    vsc_laps = set()
    if np.random.rand() < MONZA_PARAMS['vsc_probability']:
        vsc_start = np.random.choice(range(10, num_laps-5))
        vsc_laps.update(range(vsc_start, vsc_start + 2))
    
    for stint_idx, stint in enumerate(strategy):
        comp = stint["compound"]
        stint_len = stint["laps"]
        
        remaining_laps = num_laps - current_lap + 1
        stint_len = min(stint_len, remaining_laps)
        
        for lap_in_stint in range(1, stint_len + 1):
            if current_lap > num_laps:
                break
            
            current_weather = 'dry'
            track_evolution = track_evolution_rate * current_lap
            
            # USE PRACTICE-BASED TIRE MODELS HERE (same as Zandvoort)
            lap_time = get_tire_performance_from_practice_models(
                comp, lap_in_stint, compound_models, base_pace, 
                current_weather, track_evolution, EXTRACTED_PARAMS
            )
            
            # Car performance factor
            lap_time *= car_performance_factor
            
            # Fuel effect 
            fuel_effect = -(fuel_load * MONZA_PARAMS['fuel_effect_per_kg'])
            lap_time += fuel_effect
            fuel_load = max(5, fuel_load - fuel_consumption_rate)
            
            # Position penalty
            position_penalty = get_position_penalty(current_position, current_lap, num_laps)
            lap_time += position_penalty
            
            # DRS effect 
            if current_weather == 'dry' and current_position > 1:
                drs_usage_prob = get_drs_usage_probability()
                position_adjusted_prob = drs_usage_prob * max(0.3, 1.0 - (current_position * 0.05))
                
                if np.random.rand() < position_adjusted_prob:
                    drs_advantage = get_drs_advantage()
                    lap_time -= drs_advantage
            
            # Tire temperature effects 
            if current_weather == 'dry':
                if comp == 'SOFT' and lap_in_stint > 8:
                    lap_time += min(0.6, (lap_in_stint - 8) * 0.025)
                elif comp == 'HARD' and lap_in_stint < 6:
                    lap_time += max(0, (6 - lap_in_stint) * 0.12)
            
            # Driver error 
            base_error_rate = get_driver_error_rate('dry')
            error_probability = base_error_rate + (lap_in_stint * 0.0005)
            
            if np.random.rand() < error_probability:
                lap_time += np.random.uniform(1.2, 3.5)
            
            # Random variation
            lap_time += np.random.normal(0, 0.35)
            
            # Safety car effect
            if current_lap in sc_laps:
                lap_time *= 1.35
            elif current_lap in vsc_laps:
                lap_time *= 1.18
            
            race_time += lap_time
            current_lap += 1
            # Pit stop execution (same as Zandvoort)
        if stint_idx < len(strategy) - 1:
            pit_execution = np.random.normal(pit_time_loss, 1.2)
            
            # Reduced pit loss during SC/VSC
            if any(lap in sc_laps for lap in range(max(1, current_lap-2), current_lap+1)):
                pit_execution *= 0.20
            elif any(lap in vsc_laps for lap in range(max(1, current_lap-2), current_lap+1)):
                pit_execution *= 0.55
            
            # Tire change complexity
            prev_compound = strategy[stint_idx]["compound"]
            next_compound = strategy[stint_idx + 1]["compound"]
            if prev_compound != next_compound:
                pit_execution += 0.4
            
            race_time += max(13, pit_execution)
            
            # Strategic position changes (enhanced for Monza overtaking)
            strategy_aggressiveness = len([s for s in strategy if s["compound"] in ["SOFT"]])
            undercut_probability = 0.25 if stint_idx == 0 else 0.12
            
            position_change = 0
            if np.random.rand() < undercut_probability:
                position_change = np.random.choice([-2, -1], p=[0.25, 0.75])
            else:
                position_change = np.random.choice([-1, 0, 1, 2], p=[0.10, 0.50, 0.30, 0.10])
            
            if strategy_aggressiveness >= 2:
                position_change += np.random.choice([-1, 0], p=[0.3, 0.7])
            
            current_position = max(1, min(20, current_position + position_change))
    
    return race_time, current_position

# dry tire strategies 
dry_strategies = {
    # 1-Stop Strategies
    "1-stop (M-H)": [
        {"compound": "MEDIUM", "laps": 19},
        {"compound": "HARD", "laps": 32}
    ],
    "1-stop (H-M)": [
        {"compound": "HARD", "laps": 28},
        {"compound": "MEDIUM", "laps": 24}
    ],
    "1-stop (M-S)": [
        {"compound": "MEDIUM", "laps": 33},
        {"compound": "SOFT", "laps": 20}
    ],
    "1-stop (S-M)": [
        {"compound": "SOFT", "laps": 19},
        {"compound": "MEDIUM", "laps": 34}
    ],
    "1-stop (S-H)": [
        {"compound": "SOFT", "laps": 19},
        {"compound": "HARD", "laps": 34}
    ],
    
    # 2-Stop Strategies
    "2-stop (M-H-M)": [
        {"compound": "MEDIUM", "laps": 13},
        {"compound": "HARD", "laps": 19},
        {"compound": "MEDIUM", "laps": 19}
    ],
    "2-stop (M-H-S)": [
        {"compound": "MEDIUM", "laps": 16},
        {"compound": "HARD", "laps": 29},
        {"compound": "SOFT", "laps": 7}
    ],
    "2-stop (S-M-S)": [
        {"compound": "SOFT", "laps": 19},
        {"compound": "MEDIUM", "laps": 22},
        {"compound": "SOFT", "laps": 13}
    ],
    "2-stop (S-H-S)": [
        {"compound": "SOFT", "laps": 23},
        {"compound": "HARD", "laps": 24},
        {"compound": "SOFT", "laps": 6}
    ],
    
    # 3-Stop Strategies
    "3-stop (M-S-H-S)": [
        {"compound": "MEDIUM", "laps": 24},
        {"compound": "SOFT", "laps": 4},
        {"compound": "HARD", "laps": 11},
        {"compound": "SOFT", "laps": 3}
    ]
}

all_strategies = dry_strategies

# Convenience functions for extracted parameters 
def get_tire_offset(compound):
    if PARAMETERS_LOADED:
        return TIRE_PERFORMANCE.get(compound, {}).get('offset', 0.0)
    else:
        offsets = {'SOFT': 0.0, 'MEDIUM': 0.37, 'HARD': 0.70}
        return offsets.get(compound, 0.0)

def get_tire_degradation_rate(compound):
    if PARAMETERS_LOADED:
        return TIRE_PERFORMANCE.get(compound, {}).get('degradation_rate', 0.07)
    else:
        rates = {'SOFT': 0.12, 'MEDIUM': 0.07, 'HARD': 0.04}
        return rates.get(compound, 0.07)

def get_position_penalty_extracted(position):
    if PARAMETERS_LOADED:
        return POSITION_PENALTIES.get(position, {}).get('penalty', 0.025 * (position - 1))
    else:
        return 0.025 * (position - 1)

def get_driver_error_rate(weather_condition='dry'):
    if PARAMETERS_LOADED:
        return DRIVER_ERROR_RATES.get(weather_condition, {}).get('base_error_rate', 0.008)
    else:
        return 0.008

def get_drs_advantage():
    if PARAMETERS_LOADED:
        mean_adv = DRS_EFFECTIVENESS.get('median_advantage', 0.48)
        std_adv = DRS_EFFECTIVENESS.get('std_advantage', 0.22)
        return max(0.2, np.random.normal(mean_adv, std_adv))
    else:
        return 0.48

def get_drs_usage_probability():
    if PARAMETERS_LOADED:
        return DRS_EFFECTIVENESS.get('usage_probability', 0.45)
    else:
        return 0.45

# Points system (same as Zandvoort)
def get_f1_points(position):
    """Return F1 championship points for finishing position"""
    points_map = {
        1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1
    }
    return points_map.get(position, 0)
# Monte Carlo simulation (same structure as Zandvoort)
def run_monte_carlo_with_grid(strategies, compound_models, grid_positions, num_sims=500):
    """Run simulation across different grid positions with practice-based tire models"""
    results = {}
    
    # Car performance factors based on 2025 grid positions
    car_performance_map = {
        1: 0.98,   2: 0.985,  3: 0.99,   4: 0.995,  5: 1.00,   
        6: 1.005,  7: 1.01,   8: 1.015,  9: 1.02,   10: 1.025,
        11: 1.03,  12: 1.035, 13: 1.04,  14: 1.045, 15: 1.05,  
    }
    
    for grid_pos in grid_positions:
        print(f"\nSimulating from grid position {grid_pos} using practice-based tire models...")
        pos_results = {name: {'times': [], 'final_positions': [], 'points': []} for name in strategies.keys()}
        
        car_factor = car_performance_map.get(grid_pos, 1.0 + (grid_pos - 15) * 0.005)
        
        for _ in trange(num_sims, desc=f"P{grid_pos}"):
            base_pace = np.random.normal(MONZA_PARAMS['base_pace'], 0.5)
            sim_rain_prob = MONZA_PARAMS['rain_probability']
            
            for name, strat in strategies.items():
                race_time, final_pos = simulate_race(
                    strat, grid_pos, compound_models, 
                    base_pace=base_pace, rain_probability=sim_rain_prob,
                    car_performance_factor=car_factor
                )
                
                points = get_f1_points(final_pos)
                
                pos_results[name]['times'].append(race_time)
                pos_results[name]['final_positions'].append(final_pos)
                pos_results[name]['points'].append(points)
        
        results[grid_pos] = pos_results
    
    return results

# Visualization function 
def plot_strategy_analysis_with_practice_models(results, grid_positions, model_info):
    """Create comprehensive strategy analysis plots with practice model info"""
    
    rain_prob = MONZA_PARAMS['rain_probability']
    model_source = "Free Practice Models" if compound_models else "Fallback Models"
    
    # Figure 1: Race time distributions
    fig1, axes1 = plt.subplots(1, 3, figsize=(20, 7))
    fig1.suptitle(f'Italian GP Strategy Analysis - Race Time Distributions\n'
                 f'({rain_prob:.0%} Rain | {model_source})', fontsize=14)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_strategies)))
    strategy_colors = dict(zip(all_strategies.keys(), colors))
    
    for i, grid_pos in enumerate([1, 3, 5]):
        ax = axes1[i]
        for strategy_name in all_strategies.keys():
            times = results[grid_pos][strategy_name]['times']
            ax.hist(times, bins=25, alpha=0.6, label=strategy_name, 
                   color=strategy_colors[strategy_name])
        
        ax.set_title(f'Race Time Distribution - Grid P{grid_pos}')
        ax.set_xlabel('Race Time (s)')
        ax.set_ylabel('Frequency')
        if i == 2:  
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add minimal watermark
        ax.text(0.99, 0.01, 'j5t3313', transform=ax.transAxes,
               fontsize=8, color='lightgray', alpha=0.4,
               ha='right', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Figure 2: Tire model comparison (same as Zandvoort)
    if compound_models and model_info:
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6))
        
        compounds_modeled = model_info.get('compounds_modeled', [])
        
        if compounds_modeled:
            # Show tire degradation curves from practice models
            stint_laps = np.arange(1, 31)
            
            for compound in compounds_modeled:
                if compound in compound_models:
                    try:
                        samples = compound_models[compound].get_samples()
                        alpha_median = np.median(samples['alpha'])
                        beta_median = np.median(samples['beta'])
                        
                        lap_times = alpha_median + beta_median * stint_laps
                        ax2.plot(stint_laps, lap_times, label=f'{compound} (Practice Model)', 
                                marker='o', markersize=3, linewidth=2)
                    except:
                        pass
            
            ax2.set_xlabel('Stint Lap')
            ax2.set_ylabel('Predicted Lap Time (s)')
            ax2.set_title('Tire Degradation Models from Practice Data')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add model quality information
            quality_text = "Model Quality:\n"
            for compound in compounds_modeled:
                if compound in model_info.get('model_quality', {}):
                    quality = model_info['model_quality'][compound]['quality']
                    data_points = model_info['model_quality'][compound]['data_points']
                    quality_text += f"{compound}: {quality} ({data_points} laps)\n"
            
            ax2.text(0.02, 0.98, quality_text, transform=ax2.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            
            ax2.text(0.99, 0.01, 'j5t3313', transform=ax2.transAxes,
                   fontsize=8, color='lightgray', alpha=0.4,
                   ha='right', va='bottom')
            
            plt.tight_layout()
            plt.show()
            # Run simulations with practice-based tire models
key_positions = [1, 3, 5, 8, 10, 15]
print("\n" + "="*80)
print("RUNNING MONTE CARLO SIMULATION WITH PRACTICE-BASED TIRE MODELS")
print("="*80)

# Show tire model status
if compound_models:
    print(f"Using tire models built from practice sessions")
    print(f"Compounds modeled: {', '.join(model_info.get('compounds_modeled', []))}")
    print(f"Total practice laps used: {model_info.get('total_practice_laps', 0)}")
else:
    print(f"No practice models available - using fallback tire modeling")

print(f"Simulation parameters: {MONZA_PARAMS['rain_probability']:.0%} rain, "
      f"{MONZA_PARAMS['sc_probability']:.0%} SC, {MONZA_PARAMS['vsc_probability']:.0%} VSC")

results = run_monte_carlo_with_grid(all_strategies, compound_models, key_positions, num_sims=300)

# Generate analysis with practice model information
plot_strategy_analysis_with_practice_models(results, key_positions, model_info)

# Results summary
print("\n" + "="*80)
print("ITALIAN GP STRATEGY ANALYSIS SUMMARY (WITH PRACTICE MODELS)")
print("="*80)

tire_model_source = "Practice Models" if compound_models else "Fallback Models"
print(f"Tire Models: {tire_model_source}")

if compound_models and model_info:
    print(f"Practice Data Sources:")
    session_info = model_info.get('session_info', {})
    if session_info.get('fp1_available'):
        print(f"  FP1: {session_info.get('fp1_laps', 0)} laps")
    if session_info.get('fp2_available'):
        print(f"  FP2: {session_info.get('fp2_laps', 0)} laps")
    
    print(f"Model Quality:")
    for compound, quality_info in model_info.get('model_quality', {}).items():
        quality = quality_info['quality']
        data_points = quality_info['data_points']
        print(f"  {compound}: {quality} quality ({data_points} practice laps)")

print(f"Extracted Parameters: {'Yes' if PARAMETERS_LOADED else 'Fallback values'}")

for grid_pos in key_positions:
    print(f"\nGRID POSITION {grid_pos}")
    print("-" * 50)
    
    strategy_summary = []
    for strategy_name, data in results[grid_pos].items():
        times = np.array(data['times'])
        positions = np.array(data['final_positions'])
        points = np.array(data['points'])
        
        strategy_summary.append({
            'Strategy': strategy_name,
            'Avg Time': f"{np.mean(times):.1f}s",
            'Avg Final Pos': f"{np.mean(positions):.1f}",
            'Avg Points': f"{np.mean(points):.1f}",
            'Points Finish %': f"{np.mean(points > 0)*100:.1f}%",
            'Top 5 Finish %': f"{np.mean(positions <= 5)*100:.1f}%",
            'Podium %': f"{np.mean(positions <= 3)*100:.1f}%" if grid_pos <= 10 else "N/A",
            'Win %': f"{np.mean(positions == 1)*100:.1f}%" if grid_pos <= 5 else "N/A"
        })
    
    summary_df = pd.DataFrame(strategy_summary)
    summary_df = summary_df.sort_values('Avg Points', ascending=False)
    print(summary_df.to_string(index=False))

print("\n" + "="*80)
print("KEY IMPROVEMENTS WITH PRACTICE-BASED MODELING")
print("="*80)

if compound_models:
    print("ADVANTAGES OF FP1+FP2 TIRE MODELS:")
    print("  More accurate tire degradation rates from actual track data")
    print("  Accounts for Monza-specific track surface characteristics") 
    print("  Includes both morning (FP1) and afternoon (FP2) track conditions")
    print("  Better representation of compound performance differences")
    print("  Reduced dependency on generic tire modeling assumptions")
    
    if model_info.get('session_info', {}).get('fp1_available') and model_info.get('session_info', {}).get('fp2_available'):
        print("  Full practice data coverage - both FP1 AND FP2 available")
    elif model_info.get('session_info', {}).get('fp1_available'):
        print("  FP1 data available")
    elif model_info.get('session_info', {}).get('fp2_available'):
        print("  FP2 data available")
else:
    print("PRACTICE MODEL LIMITATIONS:")
    print("  No FP1/FP2 data available for tire modeling")
    print("  Using fallback tire degradation estimates")
    print("  May not capture Monza-specific tire behavior")
    print("  Consider running closer to race weekend for better data")

print("\nRECOMMENDATION:")
if compound_models:
    print("Use these practice-based results for strategy planning - they incorporate")
    print("actual Monza tire behavior from FP1 and FP2 sessions.")
else:
    print("Re-run this analysis after FP1/FP2 sessions for more accurate tire modeling.")

print("\n" + "="*80)