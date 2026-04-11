import os
import pandas as pd
from main import run_pipeline
from config import DATASET_CONFIG

# data_folder = "datasets/20kpatients"
# output_folder = "LabsNew0.9Attr"

# data_folder = "datasets/20kemp"
# output_folder = "EmpNew0.9Attr"

data_folder = "datasets/20kweather"
output_folder = "WeatherNew0.9Attr"

# # diff_folder = "diffFunctions/PatientLabsSupp"

os.makedirs(output_folder, exist_ok=True)

csv_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]

for csv_file in csv_files:
    dataset_name = csv_file.split(".")[0]

    input_path = os.path.join(data_folder, csv_file)
    output_file = os.path.join(output_folder, f"{dataset_name}_pipeline_output.csv")

    # diff_file = os.path.join(diff_folder, f"{dataset_name}_diff_functions.txt")
    diff_file = None

    # DATASET_CONFIG[dataset_name] = {
    #     "file": input_path,
    #     "target_attribute": "Hemoglobin",
    #     "target_flag": ""
    # }
    # DATASET_CONFIG[dataset_name] = {
    #     "file": input_path,
    #     "target_attribute": "FIRM",
    #     "target_flag": ""
    # }
    DATASET_CONFIG[dataset_name] = {
        "file": input_path,
        "target_attribute": "Max_Temp_C",
        "target_flag": ""
    }

    print(f"\n=== Running pipeline for {csv_file} ===")
    print(f"Using diff functions: {diff_file}")

    # thetas = {0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
    thetas = {0.9}
    for theta in thetas:
        outputfile=f"{output_file}_supp_{theta}.csv"
        print(f"For min support threshold: {theta}")
        # diff_file = os.path.join(diff_folder, f"{dataset_name}_supp_{theta}_diff_functions.txt")
        
        # if not os.path.exists(diff_file):
        #     print(f"No diff file for {dataset_name}, skipping.")
        #     continue

        if os.path.exists(outputfile):
            print(f"Skipping {csv_file} (already processed)")
            continue

        run_pipeline(dataset_name, output_file=outputfile, diff_file=diff_file, givenTheta = theta)
        print(f"Results saved to {output_file}")