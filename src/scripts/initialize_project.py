import os
import argparse


# TODO:
# - Create the target_grid file in the config folder 
# - Create the settings file in the config folder




def welcome_message():
    print('\nWelcome in XSeasonsDetect project initialization script!\n')    

def create_folder_tree(base_path, folder_structure):

    for folder in folder_structure:
        # Create the full path for the folder
        path = os.path.join(base_path, folder)
        # Create the directory
        os.makedirs(path, exist_ok=True)  # `exist_ok=True` avoids raising an error if the directory already exists
        print(f"Created: {path}")

def create_proect_directories(name):

    # Define the base path
    base_path = os.path.join(os.getcwd(), name)

    # Define the folder structure
    folder_structure = [
        'data/raw/ERA5',
        'data/raw/shapefiles',
        'data/raw/reference_dates',

        'data/temp/ERA5',

        'data/preprocessed/ERA5',
        'data/preprocessed/shapefiles',
        'data/preprocessed/reference_dates',

        'results/figures',
        'results/files',

        'config',

        'notebooks',
    ]
    # Create the folder tree
    create_folder_tree(base_path, folder_structure)
    
def main():

    # Initialize the parser
    parser = argparse.ArgumentParser(description='Initialize a new XSeasonsDetect project')
    
    # Add the arguments - Name of the project
    parser.add_argument('--name', type=str, help='Name of the project', required=True)

    # Parse the arguments
    args = parser.parse_args()

    welcome_message()
    create_proect_directories(args.name)
