import os
import pandas as pd

def process_and_save_omics_data(input_dir, omics_file_names, output_dir):
    """
    Processes omics and clinical data and saves them into a single file.

    Args:
    input_dir (str): Directory containing the input data files.
    omics_file_names (list of str): Filenames of omics data files (without file extension).
    output_dir (str): Directory where the processed data file will be saved.

    Returns:
    None
    """
    # Check and prepare the output directory
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        print(f"Created output directory {output_dir}.")
    else:
        print(f"Output directory {output_dir} already exists.")

    # Load and clean clinical data
    clinical_data = pd.read_csv(os.path.join(input_dir, "clinical.csv.gz")).dropna(axis=0)
    patient_ids = set(clinical_data['sample_id'])

    # Initialize DataFrame for combined omics data
    combined_omics_data = pd.DataFrame()

    # Process and combine each omics data file
    for omics_file in omics_file_names:
        omics_data = pd.read_csv(os.path.join(input_dir, omics_file + ".csv.gz"), index_col=0).dropna(axis=1)

        # Update patient_ids with intersection
        patient_ids &= set(omics_data.index)

        # Renaming gene columns with omics data type
        omics_data.columns = [f"{gene}_{omics_file}" for gene in omics_data.columns]

        # Print the number of genes in current omics data
        print(f"Number of genes in {omics_file}: {len(omics_data.columns)}")

        # Join with combined omics data
        if combined_omics_data.empty:
            combined_omics_data = omics_data
        else:
            combined_omics_data = combined_omics_data.join(omics_data, how='inner')

    # Filter combined omics data for common patient IDs
    patient_ids_list = list(patient_ids)
    combined_omics_data = combined_omics_data.loc[patient_ids_list]

    # Process and join clinical data
    clinical_data.set_index('sample_id', inplace=True)
    clinical_data_subset = clinical_data.loc[patient_ids_list, ["ostime", "status"]]
    clinical_data_subset.columns = ["OS_DAYS", "OS_EVENT"]

    # Combine omics data with clinical data
    final_dataset = combined_omics_data.join(clinical_data_subset)

    # Print the total number of samples in the final dataset
    print(f"Total number of samples in the final dataset: {final_dataset.shape[0]}")

    # Save the combined dataset to a single CSV file
    final_dataset.to_csv(os.path.join(output_dir, "data.csv"), index_label="SAMPLE_ID")

    print(f"Processed data saved in {os.path.join(output_dir, 'data.csv')}")

input_dir = "./"
output_dir = input_dir + "OmicsData/"
omics_file_names = ["rna", "scna", "methy"]
process_and_save_omics_data(input_dir, omics_file_names, output_dir)
