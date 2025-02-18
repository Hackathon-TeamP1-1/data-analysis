import os
import filecmp

def find_duplicate_csvs(folder_path):
    """
    Compare all CSV files in the given folder and return a list of pairs
    that are duplicates of each other.
    """
    # Gather all .csv files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    
    duplicates = []
    
    # Compare each file with the rest
    for i in range(len(csv_files)):
        for j in range(i + 1, len(csv_files)):
            file1 = os.path.join(folder_path, csv_files[i])
            file2 = os.path.join(folder_path, csv_files[j])
            
            # Compare contents (not just metadata) by using shallow=False
            if filecmp.cmp(file1, file2, shallow=False):
                duplicates.append((csv_files[i], csv_files[j]))
    
    return duplicates

if __name__ == "__main__":
    # Example usage: check for duplicates in the "2024" folder
    folder_to_check = "data/nasa/2025/"  # change this to the path of your 2024 folder
    
    # Find duplicates
    dup_list = find_duplicate_csvs(folder_to_check)
    
    # Print results
    if dup_list:
        print(f"Duplicate files found in folder '{folder_to_check}':")
        for dup_pair in dup_list:
            print(f"  {dup_pair[0]}  <-->  {dup_pair[1]}")
    else:
        print(f"No duplicates found in folder '{folder_to_check}'.")
