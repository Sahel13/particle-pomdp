#!/usr/bin/env python3
"""
Script to extract mean and std from a multi-seed experiments and save to CSV.
"""

import argparse
import pandas as pd
import wandb
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Extract experiment statistics')
    parser.add_argument('--team', type=str, required=True, help='Wandb team name')
    parser.add_argument('--project', type=str, required=True, help='Wandb project name')
    parser.add_argument('--group', type=str, help='Experiment group name (optional, will prompt if not provided)')
    parser.add_argument('--smooth', action='store_true', help='Apply running average smoothing')
    parser.add_argument('--window', type=int, default=10, help='Window size for running average (default: 10)')
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
    """Apply running average smoothing to the data."""
    # Create a copy to avoid modifying the original
    smoothed_df = df.copy()

    # Apply smoothing to each seed's data separately
    for seed in smoothed_df['seed'].unique():
        seed_data = smoothed_df[smoothed_df['seed'] == seed]

        # Sort by step to ensure proper smoothing
        seed_data = seed_data.sort_values('_step')

        # Apply rolling mean to expected_reward
        smoothed_values = seed_data['expected_reward'].rolling(window=window_size, min_periods=1).mean()

        # Update the original dataframe
        smoothed_df.loc[smoothed_df['seed'] == seed, 'expected_reward'] = smoothed_values

    return smoothed_df

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
    keys_to_fetch = ['_step', 'expected_reward']

    # Initialize an empty DataFrame
    combined_data = pd.DataFrame()

    # Process each run and merge directly into the combined DataFrame
    for run in tqdm(runs_list, desc="Processing runs"):
        # Use scan_history with keys parameter to fetch only the data we need
        print(f"Fetching history for run {run.id}...")
        history = run.history(keys=keys_to_fetch)
        df = pd.DataFrame(history)
        df['seed'] = run.config.get('seed', run.id)
        
        # Merge the new DataFrame with the combined DataFrame
        combined_data = pd.concat([combined_data, df], ignore_index=True)
    
    print(f"Total data points: {len(combined_data)}")

    # Apply smoothing if requested
    if args.smooth:
        print(f"Applying running average smoothing with window size {args.window}")
        combined_data = apply_smoothing(combined_data, args.window)

    # Calculate statistics using groupby (much more efficient)
    stats_df = combined_data.groupby('_step')['expected_reward'].agg(['mean', 'std']).reset_index()
    stats_df.columns = ['Step', 'Mean', 'Std']

    # Sort by step
    stats_df = stats_df.sort_values('Step')

    # Save to CSV
    output_file = f"{group}.csv"
    stats_df.to_csv(output_file, index=False)
    print(f"CSV file saved to {output_file}")

if __name__ == "__main__":
    main()