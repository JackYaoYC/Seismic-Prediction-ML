"""
Seismic Feature Extraction Pipeline
-----------------------------------
This script processes earthquake catalog data to extract statistical and physical features
for earthquake prediction research using sliding window analysis.

Original Logic: MATLAB implementation by [Xi Wang]
Ported to Python: [2025-12-26]

Dependencies:
    - numpy
    - pandas
    - scipy (optional, but good for statistical functions)

Usage:
    Ensure your input data does not have headers and follows the column structure:
    [Year, Month, Day, Hour, Minute, Second, Lat, Lon, Depth, Mag]
"""

import numpy as np
import pandas as pd
import os
import math
from datetime import datetime

# ==========================================
# CONFIGURATION (User Settings)
# ==========================================
CONFIG = {
    'input_file': './data/Californiadata.dat',  # Path to input data
    'output_file': './data/California_features.csv',  # Path to output features
    'Mc': 3.0,  # Magnitude of completeness
    'dt': 30,  # Step size (days)
    'time_window': 730,  # Lookback window (days, 2 years)
    'time_fore': 365,  # Prediction window (days, 1 year)
    'ref_date': datetime(1920, 1, 1),  # Reference date for time calculation
    'columns': [  # Input file column mapping
        'year', 'month', 'day', 'hour', 'minute', 'second',
        'lat', 'lon', 'depth', 'mag'
    ]
}


def load_and_preprocess(filepath, cols, mc, ref_date):
    """
    Loads earthquake catalog and preprocesses time and magnitude.

    Args:
        filepath (str): Path to the .dat file.
        cols (list): List of column names.
        mc (float): Magnitude of completeness threshold.
        ref_date (datetime): Reference date for calculating relative days.

    Returns:
        pd.DataFrame: Processed dataframe with 'days_relative' column.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: {filepath}")

    # Load data (assuming whitespace separated)
    df = pd.read_csv(filepath, delim_whitespace=True, header=None, names=cols)

    # Filter by Magnitude of Completeness
    df = df[df['mag'] >= mc].copy()

    # Create a datetime column
    df['timestamp'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute', 'second']])

    # Calculate days relative to reference date (replicating Julian Day logic)
    # Note: simple subtraction gives timedelta, we convert to days (float)
    df['days_relative'] = (df['timestamp'] - ref_date).dt.total_seconds() / (24 * 3600)

    return df.reset_index(drop=True)


def calculate_gr_parameters(magnitudes, mc, bin_width=0.1):
    """
    Calculates Gutenberg-Richter parameters (a-value, b-value) using
    both Least Squares (LSQ) and Maximum Likelihood (MLK) methods.
    """
    if len(magnitudes) == 0:
        return {k: 0.0 for k in ['b_lsq', 'a_lsq', 'b_std_lsq', 'std_gr_lsq',
                                 'b_mlk', 'a_mlk', 'b_std_mlk', 'std_gr_mlk', 'dM_lsq', 'dM_mlk']}

    mag_mean = np.mean(magnitudes)
    mag_max = np.max(magnitudes)
    count = len(magnitudes)

    # --- Prepare bins for LSQ ---
    # Create bins from Mc to max_mag
    bins = np.arange(mc, mag_max + bin_width + 1e-9, bin_width)
    hist, _ = np.histogram(magnitudes, bins=bins)

    # Cumulative frequency (N >= M)
    # We iterate through the bins centers (mag_int in original code)
    mag_int = bins[:-1]
    # Calculate cumulative number of events >= mag
    num_m = np.array([np.sum(magnitudes >= m) for m in mag_int])

    # Filter out zeros to avoid log(0) error
    valid_mask = num_m > 0
    mag_int = mag_int[valid_mask]
    num_m = num_m[valid_mask]
    len_m = len(mag_int)

    if len_m < 2:  # Not enough points for regression
        return {k: 0.0 for k in ['b_lsq', 'a_lsq', 'b_std_lsq', 'std_gr_lsq',
                                 'b_mlk', 'a_mlk', 'b_std_mlk', 'std_gr_mlk', 'dM_lsq', 'dM_mlk']}

    log_num_m = np.log10(num_m)

    # --- Method 1: Least Squares (LSQ) ---
    # Implementing the specific formula from the original MATLAB code for exact reproduction
    sum_x = np.sum(mag_int)
    sum_y = np.sum(log_num_m)
    sum_xy = np.sum(mag_int * log_num_m)
    sum_xx = np.sum(mag_int * mag_int)

    b_lsq = (len_m * sum_xy - sum_x * sum_y) / (sum_x * sum_x - len_m * sum_xx)
    a_lsq = np.sum(log_num_m + b_lsq * mag_int) / len_m

    # Standard deviation calculation for LSQ
    term = (mag_int - mag_mean) ** 2
    b_std_lsq = 2.3 * (b_lsq ** 2) * np.sqrt(np.sum(term) / len_m / (len_m - 1))

    # Goodness of fit (Standard deviation of residuals)
    residuals = log_num_m - a_lsq - b_lsq * mag_int
    std_gr_lsq = np.sum(residuals ** 2) / (len_m - 1)

    # --- Method 2: Maximum Likelihood (MLK/Aki's formula) ---
    b_mlk = np.log10(np.exp(1)) / (mag_mean - mc)
    a_mlk = np.log10(count) + b_mlk * mc
    b_std_mlk = 2.3 * (b_mlk ** 2) * np.sqrt(np.sum(term) / len_m / (len_m - 1))

    residuals_mlk = log_num_m - a_mlk - b_mlk * mag_int
    std_gr_mlk = np.sum(residuals_mlk ** 2) / (len_m - 1)

    # Magnitude Deficit
    # Check for division by zero
    dM_lsq = mag_max - (a_lsq / b_lsq) if b_lsq != 0 else 0
    dM_mlk = mag_max - (a_mlk / b_mlk) if b_mlk != 0 else 0

    return {
        'b_lsq': b_lsq, 'a_lsq': a_lsq, 'b_std_lsq': b_std_lsq, 'std_gr_lsq': std_gr_lsq,
        'b_mlk': b_mlk, 'a_mlk': a_mlk, 'b_std_mlk': b_std_mlk, 'std_gr_mlk': std_gr_mlk,
        'dM_lsq': dM_lsq, 'dM_mlk': dM_mlk
    }


def calculate_seismicity_change(window_df, t_end, t_window):
    """
    Calculates Z-value and Beta-value for seismicity rate changes.
    Splits the window into two halves.
    """
    t_start = t_end - t_window
    t_mid = t_end - 0.5 * t_window
    t_bin = 0.05 * t_window  # Bin size for standard deviation calc

    # Split data
    part1 = window_df[(window_df['days_relative'] >= t_start) & (window_df['days_relative'] < t_mid)]
    part2 = window_df[(window_df['days_relative'] >= t_mid) & (window_df['days_relative'] < t_end)]

    n1 = len(part1)
    n2 = len(part2)

    # Time durations
    tw_half = 0.5 * t_window

    # Rate and Std Dev for Part 1
    r1 = n1 / tw_half
    # Histogram for std dev (using numpy histogram)
    bins1 = np.arange(t_start, t_mid + 1e-9, t_bin)
    hist1, _ = np.histogram(part1['days_relative'], bins=bins1)
    s1 = np.std(hist1) if len(hist1) > 0 else 0

    # Rate and Std Dev for Part 2
    r2 = n2 / tw_half
    bins2 = np.arange(t_mid, t_end + 1e-9, t_bin)
    hist2, _ = np.histogram(part2['days_relative'], bins=bins2)
    s2 = np.std(hist2) if len(hist2) > 0 else 0

    # Z-value
    denom = np.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2) if (n1 > 0 and n2 > 0) else 0
    z_value = (r1 - r2) / denom if denom != 0 else 0

    # Beta value (Matthews & Reasenberg)
    n_eq1 = np.sum(hist1)  # Should equal n1 roughly
    n_bin1 = len(hist1)
    winlen_days = tw_half / t_bin  # This seems to be logical number of bins

    # Avoiding division by zero
    if n_bin1 == 0: n_bin1 = 1

    f_norm = (winlen_days / n_bin1) if n_bin1 > 0 else 0

    # Original logic approximation
    # Note: There is some specific logic in original code about 'winlen_days' that might need tuning
    # Here we follow the logic: fNormInvalLength = winlen_days / nBin1

    denom_beta = np.sqrt(n_eq1 * f_norm * (1 - f_norm)) if (n_eq1 > 0 and f_norm < 1) else 1
    beta = (n2 - n_eq1 * f_norm) / denom_beta

    return z_value, beta


def get_elapsed_time(df, current_time, mag_threshold, default_year=1920):
    """Calculates time elapsed since last earthquake of magnitude >= threshold."""
    # Look at all history before current time
    subset = df[(df['days_relative'] < current_time) & (df['mag'] >= mag_threshold)]

    if len(subset) > 0:
        last_event_time = subset['days_relative'].max()
        return current_time - last_event_time
    else:
        # Default value if no event found (Original code used 1920)
        return default_year


def main():
    print(">>> Starting Seismic Feature Extraction...")

    # 1. Load Data
    df = load_and_preprocess(
        CONFIG['input_file'],
        CONFIG['columns'],
        CONFIG['Mc'],
        CONFIG['ref_date']
    )
    print(f"Loaded {len(df)} events >= Mc {CONFIG['Mc']}")

    # 2. Setup Loop
    cat_jd = df['days_relative'].values
    start_time = cat_jd[0] + CONFIG['time_window']
    end_time = cat_jd[-1] - CONFIG['time_fore']

    # Create time steps
    time_steps = np.arange(start_time, end_time, CONFIG['dt'])

    features_list = []

    print(f"Processing {len(time_steps)} time windows...")

    for i, t_curr in enumerate(time_steps):
        # Define Window
        t_start = t_curr - CONFIG['time_window']

        # Get data in window
        window_mask = (df['days_relative'] >= t_start) & (df['days_relative'] < t_curr)
        sub_df = df[window_mask]

        # --- Feature Calculation ---

        # 1. Basic Stats
        num_events = len(sub_df)
        mag_max = sub_df['mag'].max() if num_events > 0 else 0
        mag_mean = sub_df['mag'].mean() if num_events > 0 else 0

        # 2. G-R Parameters (LSQ & MLK)
        gr_params = calculate_gr_parameters(sub_df['mag'].values, CONFIG['Mc'])

        # 3. Energy (Benioff Strain)
        # Formula: sqrt(sum(10^(12 + 1.8*M))) -> based on user provided code
        energy = np.sqrt(np.sum(10 ** (12 + 1.8 * sub_df['mag']))) if num_events > 0 else 0

        # 4. Probability > M7 (Poisson)
        prob_x7_lsq = math.exp(-3 * gr_params['b_lsq'] / math.log10(math.e)) if gr_params['b_lsq'] != 0 else 0
        prob_x7_mlk = math.exp(-3 * gr_params['b_mlk'] / math.log10(math.e)) if gr_params['b_mlk'] != 0 else 0

        # 5. Seismicity Rate Changes (Z, Beta)
        z_val, beta_val = calculate_seismicity_change(sub_df, t_curr, CONFIG['time_window'])

        # 6. Elapsed Times
        # Note: We pass the FULL dataframe to search history beyond the window if needed?
        # The original code searches `cat_jd < t(i)`, which implies full history up to now.
        full_history_mask = df['days_relative'] < t_curr
        df_history = df[full_history_mask]
        t_elaps_60 = get_elapsed_time(df_history, t_curr, 6.0)
        t_elaps_65 = get_elapsed_time(df_history, t_curr, 6.5)
        t_elaps_70 = get_elapsed_time(df_history, t_curr, 7.0)
        t_elaps_75 = get_elapsed_time(df_history, t_curr, 7.5)

        # --- Label Generation (Ground Truth) ---
        # Look forward Tfore days
        fore_mask = (df['days_relative'] >= t_curr) & (df['days_relative'] < t_curr + CONFIG['time_fore'])
        fore_df = df[fore_mask]

        mag_max_obs = fore_df['mag'].max() if len(fore_df) > 0 else 0

        # Collect Row
        row = {
            'Time(Jdays)': t_curr,
            'N.O.': num_events,
            'Mag_max': mag_max,
            'Mag_mean': mag_mean,
            **gr_params,  # Unpack dictionary
            'Energy': energy,
            'x7_lsq': prob_x7_lsq,
            'x7_mlk': prob_x7_mlk,
            'zvalue': z_val,
            'beta': beta_val,
            'T_elaps6': t_elaps_60,
            'T_elaps65': t_elaps_65,
            'T_elaps7': t_elaps_70,
            'T_elaps75': t_elaps_75,
            'Mag_max_obs': mag_max_obs
        }

        features_list.append(row)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} windows...")

    # 3. Save Output
    results_df = pd.DataFrame(features_list)

    # Reorder columns to match logical grouping if desired, or just save
    results_df.to_csv(CONFIG['output_file'], index=False)
    print(f"\n>>> Success! Features saved to {CONFIG['output_file']}")
    print(results_df.head())


if __name__ == "__main__":
    main()