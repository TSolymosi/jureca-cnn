import pandas as pd
import numpy as np
import os

# Load the predictions
df = pd.read_csv("analysis_summary.csv")

# If not all thermal simulations ran through properly:
if os.path.exists("thermal_parameters.csv"):
    thermal_df = pd.read_csv("thermal_parameters.csv")
    
    rounding_rules = {
        "Tlow": 2,              # Tlow
        "abunch3cn": 3          # abunch3cn in scientific notation later
    }

    for col, digits in rounding_rules.items():
        if "NCH3CN" in col:
            df[col] = df[col].apply(lambda x: float(f"{x:.{digits}e}"))
        else:
            df[col] = df[col].round(digits)

    molecular_keys = ["Tlow",
                    "Thigh",
                    "abunch3cn",
                    "vin",
                    "incl",
                    "phi",
                    "full_finished",
                    "full_error"]
                    
    # Take the thermal parameters from the existing file and add the molecular parameters from the new predictions
    # so it looks like this df_molecular = df_thermal + df[molecular_keys]
    df_molecular = thermal_df.copy()
    df_molecular[molecular_keys] = df[molecular_keys]
    df_molecular["full_finished"] = False  # Initialize the 'full_finished' column to False
    df_molecular["full_error"] = None  # Initialize the 'full_error' column
    df_molecular.to_csv("all_parameters.csv", index=False)  # Save the DataFrame to a CSV file
    
else:
    print("thermal_parameters.csv does not exist")

    # Define rounding precision for each parameter
    rounding_rules = {
        "dens": 0,              # dens
        "lum": 0,               # lum
        "ro": 0,                # ro
        "radius": 0,            # radius
        "prho": 3,              # prho
        "Tlow": 2,              # Tlow
        "abunch3cn": 3          # abunch3cn in scientific notation later
    }

    # Apply rounding
    for col, digits in rounding_rules.items():
        if "NCH3CN" in col:
            df[col] = df[col].apply(lambda x: float(f"{x:.{digits}e}"))
        else:
            df[col] = df[col].round(digits)

    thermal_keys = ["dens",
                    "mass",
                    "lum",
                    "ri",
                    "ro",
                    "radius",
                    "prho",
                    "r_dev",
                    "phi_dev",
                    "nphot",
                    "env_disk_ratio",
                    "lines_mode",
                    "ncores",
                    "finished",
                    "error"]

    df_thermal = df[thermal_keys].copy()
    df_thermal["finished"] = False  # Initialize the 'finished' column to False
    df_thermal["error"] = None  # Initialize the 'error' column to None

    df_thermal.to_csv("thermal_parameters.csv", index=False)  # Save the DataFrame to a CSV file

    # Setting up all_parameters.csv
    df["full_finished"] = False  # Initialize the 'full_finished' column to False
    df["full_error"] = None  # Initialize the 'full_error' column to None
    df.drop(["match_score", "source"], axis=1, inplace=True)  # Drop unnecessary columns
    df.to_csv("all_parameters.csv", index=False)  # Save the DataFrame to a CSV file