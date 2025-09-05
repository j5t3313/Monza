# F1 Strategy Analysis & Monte Carlo Simulation

A comprehensive Formula 1 race strategy simulation framework that combines real telemetry data with Bayesian modeling to predict optimal pit strategies and race outcomes.

## Features

- **Practice Session Tire Modeling**: Builds tire degradation models from FP1/FP2 data using Bayesian inference
- **Circuit-Specific Parameter Extraction**: Automatically extracts track-specific parameters from historical FastF1 data
- **Monte Carlo Race Simulation**: Runs thousands of race simulations with realistic variability
- **Strategy Optimization**: Compares multiple pit strategies across different grid positions
- **Model Validation**: Validates predictions against actual race results

## Architecture

### Core Components

1. **Parameter Extraction** (`F1_Parameter_Extractor.py`)
   - Extracts circuit-specific parameters from historical race data
   - Generates position penalties, tire performance, driver error rates, and DRS effectiveness

2. **Practice Session Tire Modeling** (`fp1_fp2_tire_model_monza.py`)
   - Builds Bayesian tire degradation models from FP1/FP2 sessions
   - Uses JAX/NumPyro for efficient MCMC sampling
   - Provides compound-specific degradation curves

3. **Race Simulation Engine** (`stochasticPitSimv4_Monza.py`)
   - Monte Carlo simulation with realistic race conditions
   - Models safety cars, VSC, weather changes, and driver errors
   - Incorporates fuel effects, tire temperatures, and position-based penalties

4. **Model Validation** (`monzaStrategyModelValidation.py`)
   - Compares simulation predictions against actual race results
   - Provides accuracy metrics and visualization

## Quick Start

### Prerequisites

```bash
pip install fastf1 pandas numpy jax numpyro matplotlib seaborn scipy tqdm
```

### Basic Usage

1. **Extract Circuit Parameters**
```python
python F1_Parameter_Extractor.py 'Italian Grand Prix'
```

2. **Run Strategy Analysis**
```python
python stochasticPitSimv4_Monza.py
```

3. **Validate Model (after race)**
```python
python monzaStrategyModelValidation.py
```

## Detailed Workflow

### 1. Parameter Extraction

The system automatically extracts track-specific parameters from historical FastF1 data:

- **Position Penalties**: Traffic/dirty air effects by grid position
- **Tire Performance**: Compound offsets and degradation rates
- **Driver Error Rates**: Weather-dependent error probabilities
- **DRS Effectiveness**: Time advantage and usage probability

```python
extractor = F1ParameterExtractor('Italian Grand Prix', years=[2022, 2023, 2024])
parameters = extractor.extract_all_parameters()
```

### 2. Practice-Based Tire Modeling

Uses FP1 and FP2 session data to build compound-specific tire models:

```python
compound_models, model_info = build_compound_models_from_practice(
    year=2025, 
    gp_name='Italian Grand Prix', 
    base_pace=81.5
)
```

The system uses Bayesian linear regression to model tire degradation:
- **Priors**: Compound-specific expectations (SOFT degrades faster than HARD)
- **Likelihood**: Normal distribution around linear degradation
- **Posterior**: MCMC sampling provides uncertainty quantification

### 3. Race Simulation

Simulates complete races with multiple stochastic elements:

- **Safety Cars**: Track-specific probability with realistic timing
- **Weather**: Rain probability and tire compound changes
- **Driver Errors**: Position and stint-dependent error rates
- **Strategic Elements**: Undercuts, overcuts, and position changes

```python
results = run_monte_carlo_with_grid(
    strategies=all_strategies,
    compound_models=compound_models,
    grid_positions=[1, 3, 5, 8, 10, 15],
    num_sims=1000
)
```

### 4. Strategy Definitions

Supports multiple strategy types:

```python
strategies = {
    "1-stop (M-H)": [
        {"compound": "MEDIUM", "laps": 19},
        {"compound": "HARD", "laps": 34}
    ],
    "2-stop (S-M-S)": [
        {"compound": "SOFT", "laps": 19},
        {"compound": "MEDIUM", "laps": 22},
        {"compound": "SOFT", "laps": 12}
    ]
}
```

## Circuit Customization

The framework is designed for easy adaptation to different circuits:

1. **Update Base Parameters**:
```python
CIRCUIT_PARAMS = {
    'base_pace': 81.5,      # Typical lap time
    'num_laps': 53,         # Race distance
    'rain_probability': 0.0, # Weather likelihood
    'sc_probability': 0.35,  # Safety car probability
    'pit_time_loss': 23.7   # Pit lane time loss
}
```

2. **Extract Circuit-Specific Parameters**:
```bash
python F1_Parameter_Extractor.py 'Monaco Grand Prix'
```

3. **Update Strategy Definitions**: Modify stint lengths based on circuit characteristics

## Model Validation

The validation system compares predictions against actual race results:

- **Position Accuracy**: Mean Absolute Error in final positions
- **Strategy Effectiveness**: Points scoring and podium probabilities  
- **Condition Predictions**: Weather and safety car occurrence
- **Parameter Quality**: Sample sizes and confidence intervals

```python
validator = F1ModelValidator()
validator.load_actual_race_data(2025, 'Italian Grand Prix')
validation_results = validator.validate_position_predictions()
```

## Advanced Features

### Bayesian Tire Modeling

The tire degradation models use informative priors based on compound characteristics:

```python
def tire_model(x, y=None):
    # Base pace prior
    alpha = numpyro.sample("alpha", dist.Normal(base_pace, 2.0))
    
    # Compound-specific degradation prior
    if compound_name == 'SOFT':
        beta_prior = dist.Normal(0.15, 0.05)  # Faster degradation
    elif compound_name == 'HARD':
        beta_prior = dist.Normal(0.04, 0.02)  # Slower degradation
    
    beta = numpyro.sample("beta", beta_prior)
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    
    mu = alpha + beta * x
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=y)
```

### Position-Dependent Effects

The simulation models various position-dependent effects:

- **DRS Usage**: Higher probability for cars further back
- **Traffic Penalties**: Grid position-based time penalties that decrease during race
- **Fuel Effects**: Lap time improvement as fuel load decreases
- **Strategic Opportunities**: Position-dependent undercut/overcut success rates

### Weather Integration

Comprehensive weather modeling:

- **Rain Probability**: Circuit and season-specific likelihood
- **Tire Compound Changes**: Automatic switches to INTERMEDIATE/WET
- **Lap Time Effects**: Weather-dependent pace penalties
- **Strategic Impact**: Enhanced pit window opportunities

## Output Analysis

The system generates comprehensive analysis including:

- **Race Time Distributions**: Histogram plots by strategy and grid position
- **Position Probability Matrices**: Likelihood of finishing in each position
- **Points Expectations**: Expected championship points by strategy
- **Strategy Rankings**: Optimal strategies for each grid position

## File Structure

```
├── F1_Parameter_Extractor.py          # Parameter extraction from historical data
├── fp1_fp2_tire_model_monza.py        # Practice session tire modeling
├── stochasticPitSimv4_Monza.py        # Main simulation engine
├── monzaStrategyModelValidation.py    # Model validation framework
├── italian_gp_targeted_parameters.py  # Extracted circuit parameters
└── README.md                          # This file
```

## Limitations and Considerations

- **Data Dependency**: Requires FastF1 package and active internet connection
- **Circuit Specificity**: Parameters are circuit-specific and may not transfer
- **Practice Data**: Tire models require sufficient FP1/FP2 data for accuracy
- **Model Assumptions**: Linear tire degradation may not capture all behavior
- **Validation Lag**: Race validation only possible after actual race completion

## Dependencies

- `fastf1`: F1 telemetry data access
- `jax`/`numpyro`: Bayesian modeling and MCMC sampling
- `pandas`/`numpy`: Data manipulation and numerical computing
- `matplotlib`/`seaborn`: Visualization
- `scipy`: Statistical functions
- `tqdm`: Progress bars

## License

This project is intended for educational and research purposes. F1 data usage subject to FastF1 terms and conditions.
