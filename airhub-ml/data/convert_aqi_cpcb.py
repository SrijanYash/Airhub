"""
Convert AQI from 1-5 scale to Indian CPCB standard (0-500, extendable to 1000).

CPCB AQI Categories:
| AQI     | Category     | Health Impact       |
| ------- | ------------ | ------------------- |
| 0–50    | Good         | Minimal             |
| 51–100  | Satisfactory | Minor discomfort    |
| 101–200 | Moderate     | Breathing issues    |
| 201–300 | Poor         | Respiratory illness |
| 301–400 | Very Poor    | Serious effects     |
| 401–500 | Severe       | Emergency           |
| 501–1000| Hazardous    | Life-threatening    |
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime

# CPCB AQI breakpoints for pollutants (μg/m³ except CO in mg/m³)
# Format: (category, AQI_range, PM2.5, PM10, O3, NO2, SO2, CO)
CPCB_BREAKPOINTS = {
    'Good':          {'aqi': (0, 50),    'pm25': (0, 30),   'pm10': (0, 50),    'o3': (0, 50),    'no2': (0, 40),   'so2': (0, 40),   'co': (0, 1)},
    'Satisfactory':  {'aqi': (51, 100),  'pm25': (31, 60),  'pm10': (51, 100),  'o3': (51, 100),  'no2': (41, 80),  'so2': (41, 80),  'co': (1.1, 2)},
    'Moderate':      {'aqi': (101, 200), 'pm25': (61, 90),  'pm10': (101, 250), 'o3': (101, 160), 'no2': (81, 180), 'so2': (81, 380), 'co': (2.1, 10)},
    'Poor':          {'aqi': (201, 300), 'pm25': (91, 120), 'pm10': (251, 350), 'o3': (161, 200), 'no2': (181, 280),'so2': (381, 800),'co': (10.1, 17)},
    'Very Poor':     {'aqi': (301, 400), 'pm25': (121, 250),'pm10': (351, 430), 'o3': (201, 400), 'no2': (281, 400),'so2': (801, 1600),'co': (17.1, 34)},
    'Severe':        {'aqi': (401, 500), 'pm25': (251, 500),'pm10': (431, 1000),'o3': (401, 500), 'no2': (401, 1000),'so2': (1601, 3200),'co': (34.1, 50)},
}

# Extended breakpoints for Hazardous (501-1000)
HAZARDOUS_BREAKPOINTS = {
    'Hazardous_1':   {'aqi': (501, 600), 'pm25': (501, 600), 'pm10': (1001, 1200), 'o3': (501, 600), 'no2': (1001, 1200), 'so2': (3201, 4000), 'co': (50.1, 60)},
    'Hazardous_2':   {'aqi': (601, 700), 'pm25': (601, 700), 'pm10': (1201, 1400), 'o3': (601, 700), 'no2': (1201, 1400), 'so2': (4001, 4800), 'co': (60.1, 70)},
    'Hazardous_3':   {'aqi': (701, 800), 'pm25': (701, 800), 'pm10': (1401, 1600), 'o3': (701, 800), 'no2': (1401, 1600), 'so2': (4801, 5600), 'co': (70.1, 80)},
    'Hazardous_4':   {'aqi': (801, 900), 'pm25': (801, 900), 'pm10': (1601, 1800), 'o3': (801, 900), 'no2': (1601, 1800), 'so2': (5601, 6400), 'co': (80.1, 90)},
    'Hazardous_5':   {'aqi': (901, 1000),'pm25': (901, 1000),'pm10': (1801, 2000), 'o3': (901, 1000),'no2': (1801, 2000),'so2': (6401, 7200),'co': (90.1, 100)},
}

CPCB_BREAKPOINTS.update(HAZARDOUS_BREAKPOINTS)


def calculate_sub_index(concentration, pollutant_range, aqi_range):
    """
    Calculate AQI sub-index for a pollutant using linear interpolation.
    For values exceeding the highest breakpoint, extrapolate linearly.

    Formula: AQI = ((AQI_high - AQI_low) / (C_high - C_low)) * (C - C_low) + AQI_low
    """
    c_low, c_high = pollutant_range
    aqi_low, aqi_high = aqi_range

    if concentration <= c_low:
        return aqi_low

    # Extrapolate for values exceeding the breakpoint (enables AQI > 1000)
    if concentration >= c_high:
        slope = (aqi_high - aqi_low) / (c_high - c_low) if c_high != c_low else 1.0
        return aqi_high + slope * (concentration - c_high)

    return ((aqi_high - aqi_low) / (c_high - c_low)) * (concentration - c_low) + aqi_low


def get_aqi_category(aqi_value):
    """Return CPCB category name for an AQI value."""
    if aqi_value <= 50:
        return 'Good'
    elif aqi_value <= 100:
        return 'Satisfactory'
    elif aqi_value <= 200:
        return 'Moderate'
    elif aqi_value <= 300:
        return 'Poor'
    elif aqi_value <= 400:
        return 'Very Poor'
    elif aqi_value <= 500:
        return 'Severe'
    else:
        return 'Hazardous'


def calculate_cpcb_aqi(row):
    """
    Calculate CPCB-standard AQI from pollutant concentrations.
    Uses the maximum sub-index method (standard practice).
    Extrapolates for concentrations exceeding defined breakpoints (AQI can exceed 1000).

    Note: CO is converted from μg/m³ to mg/m³ (CPCB standard unit).

    Args:
        row: DataFrame row with pollutant columns (pm25, pm10, o3, no2, so2, co)

    Returns:
        tuple: (AQI_value, category, dominant_pollutant)
    """
    sub_indices = {}
    pollutants = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']

    for pollutant in pollutants:
        if pollutant not in row or pd.isna(row[pollutant]):
            continue

        concentration = row[pollutant]

        # Convert CO from μg/m³ to mg/m³ (CPCB standard)
        if pollutant == 'co':
            concentration = concentration / 1000.0  # μg/m³ → mg/m³

        max_sub_idx = 0

        # Find the appropriate breakpoint range for this concentration
        for category, ranges in CPCB_BREAKPOINTS.items():
            aqi_range = ranges['aqi']
            pollutant_range = ranges[pollutant]

            # Check if concentration falls within or below this range
            c_low, c_high = pollutant_range
            aqi_low, aqi_high = aqi_range

            if concentration <= c_high:
                # Calculate sub-index using this breakpoint (or extrapolate if below c_low)
                sub_idx = calculate_sub_index(concentration, pollutant_range, aqi_range)
                max_sub_idx = max(max_sub_idx, sub_idx)
                break
        else:
            # Concentration exceeds all breakpoints - use highest breakpoint for extrapolation
            highest_cat = list(CPCB_BREAKPOINTS.keys())[-1]
            highest_ranges = CPCB_BREAKPOINTS[highest_cat]
            sub_idx = calculate_sub_index(
                concentration,
                highest_ranges[pollutant],
                highest_ranges['aqi']
            )
            max_sub_idx = sub_idx

        sub_indices[pollutant] = max_sub_idx

    if not sub_indices:
        return 0, 'Good', 'none'

    # AQI is the maximum of all sub-indices
    aqi_value = max(sub_indices.values())
    dominant_pollutant = max(sub_indices, key=sub_indices.get)
    category = get_aqi_category(aqi_value)

    return round(aqi_value, 2), category, dominant_pollutant


def convert_1_to_5_scale(categorical_aqi):
    """
    Convert 1-5 scale AQI to CPCB standard midpoints.

    1 → Good (0-50) → 25
    2 → Satisfactory (51-100) → 75
    3 → Moderate (101-200) → 150
    4 → Poor (201-300) → 250
    5 → Very Poor/Severe (301-500) → 400
    """
    mapping = {
        1: 25,    # Good
        2: 75,    # Satisfactory
        3: 150,   # Moderate
        4: 250,   # Poor
        5: 400,   # Very Poor/Severe
    }
    return mapping.get(int(categorical_aqi), 0)


def convert_aqi_file(input_path, output_path=None, method='cpcb'):
    """
    Convert AQI values in a CSV file to CPCB standard.

    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV (default: input_cpcb.csv)
        method: 'cpcb' (calculate from pollutants) or 'scale' (convert 1-5 scale)

    Returns:
        DataFrame with converted AQI values
    """
    df = pd.read_csv(input_path)

    if method == 'scale' and 'aqi' in df.columns:
        # Convert existing 1-5 scale AQI
        df['aqi_cpcb'] = df['aqi'].apply(convert_1_to_5_scale)
        df['aqi_category'] = df['aqi_cpcb'].apply(get_aqi_category)
        print(f"Converted {len(df)} rows from 1-5 scale to CPCB standard")

    elif method == 'cpcb':
        # Calculate AQI from pollutant concentrations
        results = df.apply(calculate_cpcb_aqi, axis=1)
        df['aqi'] = [r[0] for r in results]
        df['aqi_category'] = [r[1] for r in results]
        df['dominant_pollutant'] = [r[2] for r in results]
        print(f"Calculated CPCB AQI for {len(df)} rows from pollutant data")

    # Auto-determine output path
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_cpcb{ext}"

    df.to_csv(output_path, index=False)
    print(f"Saved converted data to: {output_path}")

    # Print summary statistics
    print("\n=== AQI Summary ===")
    print(f"Min AQI: {df['aqi'].min():.2f}")
    print(f"Max AQI: {df['aqi'].max():.2f}")
    print(f"Mean AQI: {df['aqi'].mean():.2f}")
    print(f"\nCategory Distribution:")
    print(df['aqi_category'].value_counts().to_string())

    return df


def convert_all_files():
    """Convert all AQI CSV files in the datasets directory."""
    datasets_dir = os.path.join(os.path.dirname(__file__), 'datasets')

    for filename in os.listdir(datasets_dir):
        if filename.endswith('.csv') and 'aqi' in filename.lower():
            input_path = os.path.join(datasets_dir, filename)

            # Check if file has AQI column (1-5 scale) or pollutant columns
            df = pd.read_csv(input_path)

            if 'aqi' in df.columns and df['aqi'].max() <= 5:
                # Has 1-5 scale AQI
                convert_aqi_file(input_path, method='scale')
            elif all(col in df.columns for col in ['pm25', 'pm10']):
                # Has pollutant columns - calculate AQI
                convert_aqi_file(input_path, method='cpcb')
            else:
                print(f"Skipping {filename}: unknown format")


if __name__ == "__main__":
    print("=== CPCB AQI Converter ===\n")

    # Convert all files in datasets directory
    convert_all_files()

    print("\n=== Conversion Complete ===")
