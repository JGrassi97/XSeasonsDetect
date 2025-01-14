from preprocessing.netcdf_preprocess_CDO import *



def main():

    folders = os.listdir('data/raw/ERA5')

    print('The following folders will be processed:')

    # Create a nicely formatted string
    formatted_output = '\n'.join([f'    - {fold}' for fold in folders])

    # Print the formatted output
    print(formatted_output)

    target_grid_path = 'config/target_grid.txt'
    boundary_file = 'data/raw/shapefiles/boundary.gpkg'

    if not os.path.exists(target_grid_path):
        raise Exception('Target grid file not found!')
    
    if not os.path.exists(boundary_file):
        raise Exception('Boundary file not found!')

    else:
        boundary = gpd.read_file(boundary_file)


    for folder in folders:

        raw_path = f'{os.getcwd()}/data/raw/ERA5/{folder}'
        temp_path = f'{os.getcwd()}/data/temp/ERA5/{folder}'
        preprocess_path = f'{os.getcwd()}/data/preprocessed/ERA5/{folder}'

        if not os.path.exists(temp_path):
            os.mkdir(temp_path)
        
        if not os.path.exists(preprocess_path):
            os.mkdir(preprocess_path)

        if os.path.exists(f'{preprocess_path}/final.nc'):
            print(f'\n{folder} already processed, do you want to overwrite it?')
            answer = input('Y/N: ')

            if answer.lower() == 'n':
                continue

            if answer.lower() == 'y':
                os.remove(f'{preprocess_path}/final.nc')
                standard_preprocess(raw_path, temp_path, preprocess_path, 1961, 2019, target_grid_path, boundary, 15, True, 'final')
            
            else:
                print('Invalid input, skipping folder')
                continue
        
        else:
            standard_preprocess(raw_path, temp_path, preprocess_path, 1961, 2019, target_grid_path, boundary, 15, True, 'final')






