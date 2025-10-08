#!/usr/bin/env python3
"""
Script to extract mean and std from a multi-seed experiments and save to CSV.
"""

import argparse
import pandas as pd
import wandb
from tqdm import tqdm
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Extract experiment statistics')
    parser.add_argument('--team', type=str, required=True, help='Wandb team name')
    parser.add_argument('--project', type=str, required=True, help='Wandb project name')
    parser.add_argument('--group', type=str, help='Experiment group name')
    parser.add_argument('--ravg', action='store_true', help='Apply running average smoothing')
    parser.add_argument('--gauss', action='store_true', help='Apply Gaussian smoothing')
    parser.add_argument('--window', type=int, default=5, help='Window size for smoothing')
    parser.add_argument('--num_points', type=int, help='Number of points to plot')
    return parser.parse_args()


def get_available_groups(api, team, project):
    """Fetch all available groups from the specified project."""
    runs = api.runs(f"{team}/{project}")
    groups = set()

    for run in runs:
        if run.group:
            groups.add(run.group)

    return sorted(list(groups))


def select_group(groups):
    """Let the user select a group from the available groups."""
    if not groups:
        print("No groups found in the specified project.")
        return None

    print("\nAvailable groups:")
    for i, group in enumerate(groups, 1):
        print(f"{i}. {group}")

    while True:
        try:
            choice = int(input("\nSelect a group (enter the number): "))
            if 1 <= choice <= len(groups):
                return groups[choice - 1]
            else:
                print(f"Please enter a number between 1 and {len(groups)}")
        except ValueError:
            print("Please enter a valid number")


def apply_smoothing(df, window_size):
    """Apply running average smoothing to the data for all available keys except 'seed' and 'step'."""
    # Create a copy to avoid modifying the original
    smoothed_df = df.copy()

    # Apply smoothing to each seed's data separately
    for seed in smoothed_df['seed'].unique():
        # Get indices for this seed
        seed_indices = smoothed_df.index[smoothed_df['seed'] == seed]

        # Get seed data (no need to sort since data is already sorted)
        seed_data = smoothed_df.loc[seed_indices]

        # Apply rolling mean
        smoothed_values = seed_data['average_return'].rolling(window=window_size, min_periods=1).mean()

        # Update the values in the original position
        smoothed_df.loc[seed_indices, 'average_return'] = smoothed_values.values

    return smoothed_df

def apply_gaussian(df, window_size):
    """Apply Gaussian smoothing to the data for all available keys except 'seed' and 'step'.

    Gaussian smoothing computes a weighted average of the points, where the weights correspond to
    a gaussian distribution with the standard deviation specified as the smoothing parameter.
    The smoothed value is calculated for every input x value, based on the points occurring
    both before and after it.
    """
    # Create a copy to avoid modifying the original
    smoothed_df = df.copy()

    # Apply smoothing to each seed's data separately
    for seed in smoothed_df['seed'].unique():
        # Get indices for this seed
        seed_indices = smoothed_df.index[smoothed_df['seed'] == seed]

        # Get seed data (no need to sort since data is already sorted)
        seed_data = smoothed_df.loc[seed_indices]
        values = seed_data['average_return'].values

        # Create Gaussian kernel
        x = np.arange(-window_size, window_size + 1)
        kernel = np.exp(-(x**2) / (2 * window_size**2))
        kernel = kernel / kernel.sum()  # Normalize kernel

        # Apply convolution with the Gaussian kernel
        smoothed_values = np.convolve(values, kernel, mode='same')

        # Update the values in the original position
        smoothed_df.loc[seed_indices, 'average_return'] = smoothed_values

    return smoothed_df


def apply_downsampling(df, num_points):
    """Downsample the data to the specified number of points for each seed separately."""
    # Create a copy of the original dataframe
    downsampled_df = df.copy()

    # Get unique steps from the first seed to ensure consistency
    first_seed = downsampled_df['seed'].iloc[0]
    first_seed_data = downsampled_df[downsampled_df['seed'] == first_seed]
    unique_steps = first_seed_data['step'].unique()

    # Calculate which steps to keep
    positions_to_keep = np.linspace(0, len(unique_steps) - 1, num_points, dtype=int)
    steps_to_keep = unique_steps[positions_to_keep]

    # Keep only the rows with these steps for all seeds
    keep_mask = downsampled_df['step'].isin(steps_to_keep)

    # Get the downsampled dataframe
    return downsampled_df[keep_mask]

def main():
    args = parse_args()

    # Initialize wandb API
    api = wandb.Api()

    # If group is not provided, fetch available groups and let the user choose
    group = args.group
    if not group:
        groups = get_available_groups(api, args.team, args.project)
        group = select_group(groups)

        if not group:
            print("No group selected. Exiting.")
            return

    print(f"Analyzing group: {group}")

    # Get all runs from the specified team, project and group
    runs_iterator = api.runs(f"{args.team}/{args.project}", filters={"group": group})

    # Convert to list once<
    runs_list = list(runs_iterator)

    # Print the number of runs found
    print(f"Found {len(runs_list)} runs in group '{group}'")

    # Collect data from all runs more efficiently
    print("Collecting data from runs...")

    # Define the keys we need from the history
    keys_to_fetch = ['step', 'average_return']

    # Initialize an empty DataFrame
    combined_data = pd.DataFrame()

    # Process each run and merge directly into the combined DataFrame
    for run in tqdm(runs_list, desc="Processing runs"):
        # Use scan_history with keys parameter to fetch only the data we need
        print(f"Fetching history for run {run.id}...")
        history = run.scan_history(keys=keys_to_fetch)
        df = pd.DataFrame(history)
        df['seed'] = run.config.get('seed', run.id)

        # Merge the new DataFrame with the combined DataFrame
        combined_data = pd.concat([combined_data, df], ignore_index=True)

    print(f"Total data points: {len(combined_data)}")

    # Sort the combined data by step
    combined_data = combined_data.sort_values('step')

    # Apply smoothing if requested
    if args.ravg:
        print(f"Applying running average smoothing with window size {args.window}")
        combined_data = apply_smoothing(combined_data, args.window)
    elif args.gauss:
        print(f"Applying Gaussian smoothing with window size {args.window}")
        combined_data = apply_gaussian(combined_data, args.window)

    # Downsample data if num_points is specified
    if args.num_points:
        print(f"Downsampling data to {args.num_points} points")
        combined_data = apply_downsampling(combined_data, args.num_points)

    # Calculate statistics using groupby (much more efficient)
    stats_df = combined_data.groupby('step')['average_return'].agg(['mean', 'sem']).reset_index()
    stats_df.columns = ['Step', 'Mean', 'Sem']

    # Save to CSV
    output_file = f"{group}.csv"
    stats_df.to_csv(output_file, index=False)
    print(f"CSV file saved to {output_file}")


if __name__ == "__main__":
    main()