"""
Extracted F1 simulation parameters for Italian Grand Prix
Targeted extraction: Position penalties, tire performance, driver errors, DRS
Generated automatically from historical FastF1 data
"""

import numpy as np

# POSITION PENALTIES
POSITION_PENALTIES = {7: {'penalty': 0.7935036557675493, 'std': 0.2987518625920776, 'sample_size': 3}, 1: {'penalty': 0.3360419541770081, 'std': 0.3657755736700045, 'sample_size': 3}, 2: {'penalty': -0.27992358658732946, 'std': 0.10036537643505627, 'sample_size': 3}, 18: {'penalty': 2.967627537860931, 'std': 0.3645564366547273, 'sample_size': 3}, 19: {'penalty': 2.924011963723768, 'std': 0.652621170770987, 'sample_size': 3}, 13: {'penalty': 2.1076069093701313, 'std': 0.354784075150398, 'sample_size': 3}, 3: {'penalty': 0.46965404717085035, 'std': 0.7297038640405251, 'sample_size': 3}, 5: {'penalty': 0.3146847946625413, 'std': 0.1355073576414478, 'sample_size': 3}, 8: {'penalty': 0.9046813428296713, 'std': 0.22362571493949332, 'sample_size': 3}, 9: {'penalty': 1.0232581267345828, 'std': 0.023274456647686927, 'sample_size': 3}, 14: {'penalty': 1.6927943288751386, 'std': 0.09484440414726387, 'sample_size': 3}, 17: {'penalty': 2.5355344543307363, 'std': 0.40785305843125397, 'sample_size': 3}, 15: {'penalty': 2.2138932933761635, 'std': 0.3367809028835383, 'sample_size': 3}, 20: {'penalty': 2.7241200910390106, 'std': 0.4728782228484198, 'sample_size': 3}, 10: {'penalty': 1.421326184588523, 'std': 0.20393672652415523, 'sample_size': 3}, 16: {'penalty': 2.360179915616064, 'std': 0.5417430546709064, 'sample_size': 3}, 4: {'penalty': -0.017385884796090895, 'std': 0.5042757174770938, 'sample_size': 3}, 12: {'penalty': 1.595555081352862, 'std': 0.39808401566083346, 'sample_size': 3}, 6: {'penalty': 0.33064746045624466, 'std': 0.35474728401089556, 'sample_size': 3}, 11: {'penalty': 0.6470932982607261, 'std': 0.6224808672362506, 'sample_size': 3}}

# TIRE PERFORMANCE
TIRE_PERFORMANCE = {'SOFT': {'base_time': 85.928, 'degradation_rate': 0.0, 'r_squared': 0.01, 'sample_size': 154, 'offset': 0.0}, 'MEDIUM': {'base_time': 86.997, 'degradation_rate': 0.0, 'r_squared': 0.01, 'sample_size': 1043, 'offset': 1.0690000000000026}, 'HARD': {'base_time': 86.102, 'degradation_rate': 0.0, 'r_squared': 0.01, 'sample_size': 1417, 'offset': 0.1740000000000066}}

# DRIVER ERROR RATES
DRIVER_ERROR_RATES = {'dry': {'base_error_rate': 0.04, 'sample_size': 100}, 'wet': {'base_error_rate': 0.08, 'sample_size': 20}}

# DRS EFFECTIVENESS
DRS_EFFECTIVENESS = {'mean_advantage': 0.35, 'median_advantage': 0.32, 'std_advantage': 0.18, 'sample_size': 500, 'usage_probability': 0.35}

# CONVENIENCE FUNCTIONS

def get_position_penalty(position):
    """Get traffic/dirty air penalty for grid position"""
    if position in POSITION_PENALTIES:
        return POSITION_PENALTIES[position]["penalty"]
    else:
        # Extrapolate for positions beyond data
        if position <= 20:
            return 0.05 * (position - 1)  # Linear approximation
        else:
            return 1.0  # High penalty for back of grid

def get_tire_offset(compound):
    """Get tire compound offset relative to fastest"""
    return TIRE_PERFORMANCE.get(compound, {}).get("offset", 0.0)

def get_tire_degradation_rate(compound):
    """Get tire degradation rate in s/lap"""
    return TIRE_PERFORMANCE.get(compound, {}).get("degradation_rate", 0.08)

def get_driver_error_rate(weather_condition="dry"):
    """Get driver error probability per lap"""
    return DRIVER_ERROR_RATES.get(weather_condition, {}).get("base_error_rate", 0.01)

def get_drs_advantage():
    """Get DRS time advantage in seconds"""
    mean_adv = DRS_EFFECTIVENESS.get("median_advantage", 0.25)
    std_adv = DRS_EFFECTIVENESS.get("std_advantage", 0.1)
    return max(0.1, np.random.normal(mean_adv, std_adv))

def get_drs_usage_probability():
    """Get probability of being in DRS range"""
    return DRS_EFFECTIVENESS.get("usage_probability", 0.3)
