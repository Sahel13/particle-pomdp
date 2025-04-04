import argparse
import glob
import os

import pandas as pd


def combine_csv_files(folder_name: str):
    """Combine training data for multiple seeds."""
    # Get a list of all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_name, "training_log_seed_*.csv"))

    # Dictionary to store dataframes for each seed
    dfs = {}

    # Read data from each CSV file into a dataframe
    for file_path in csv_files:
        seed = os.path.basename(file_path).split("_")[-1].split(".")[0]
        df = pd.read_csv(file_path)
        df = df.rename(columns={"Average reward": f"Seed_{seed}"})
        dfs[seed] = df.set_index("Step")

    # Combine all dataframes on the 'Step' column
    combined_df = pd.concat(dfs.values(), axis=1)

    # Compute the mean and standard deviation for each step
    combined_df["Mean"] = combined_df.mean(axis=1)
    combined_df["Std"] = combined_df.std(axis=1)

    # Write the combined dataframe to a new CSV file
    combined_file_path = os.path.join(folder_name, "combined_training_log.csv")
    combined_df.to_csv(combined_file_path)

    # Write the mean and standard deviation to a separate CSV file
    summary_df = combined_df[["Mean", "Std"]]
    summary_file_path = os.path.join(folder_name, "summary_training_log.csv")
    summary_df.to_csv(summary_file_path)


parser = argparse.ArgumentParser(description="Combine CSV files.")
parser.add_argument(
    "--folder", type=str, required=True, help="Folder containing the CSV files"
)
args = parser.parse_args()

combine_csv_files(args.folder)
