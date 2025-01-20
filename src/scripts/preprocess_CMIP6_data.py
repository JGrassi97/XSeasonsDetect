from preprocessing.netcdf_preprocess_CDO import *



def main():

    models = os.listdir('data/raw/CMIP6')

    target_grid_path = 'config/target_grid.txt'
    boundary_file = 'data/raw/shapefiles/boundary.gpkg'

    if not os.path.exists(target_grid_path):
        raise Exception('Target grid file not found!')
    
    if not os.path.exists(boundary_file):
        raise Exception('Boundary file not found!')

    else:
        boundary = gpd.read_file(boundary_file)

    
    print('The following models will be processed:')

    # Create a nicely formatted string
    formatted_output = '\n'.join([f'    - {mod}' for mod in models])

    # Print the formatted output
    print(formatted_output)

    for model in models:

        scenarios = os.listdir(f'data/raw/CMIP6/{model}')

        for scenario in scenarios:

            variables = os.listdir(f'data/raw/CMIP6/{model}/{scenario}')

            for variable in variables:


                raw_path = f'{os.getcwd()}/data/raw/CMIP6/{model}/{scenario}/{variable}'
                temp_path = f'{os.getcwd()}/data/temp/CMIP6/{model}/{scenario}/{variable}'
                preprocess_path = f'{os.getcwd()}/data/preprocessed/CMIP6/{model}/{scenario}/{variable}'

                if not os.path.exists(temp_path):
                    os.makedirs(temp_path)
                
                if not os.path.exists(preprocess_path):
                    os.makedirs(preprocess_path)

                if os.path.exists(f'{preprocess_path}/final.nc'):

                    # print(f'\n{model} - {scenario} - {variable} already processed, do you want to overwrite it?')
                    # answer = input('Y/N: ')

                    # if answer.lower() == 'n':
                    #     continue

                    # if answer.lower() == 'y':
                    #     os.remove(f'{preprocess_path}/final.nc')

                    #     if scenario == 'historical':
                    #         standard_preprocess(raw_path, temp_path, preprocess_path, 1950, 2015, target_grid_path, boundary, 15, True, 'final')
                        
                    #     else:
                    #         standard_preprocess(raw_path, temp_path, preprocess_path, 2014, 2100, target_grid_path, boundary, 15, True, 'final')
                    
                    # else:
                    #     print('Invalid input, skipping folder')
                    #     continue

                    continue
                
                else:
                    
                    if scenario == 'historical':
                        standard_preprocess(raw_path, temp_path, preprocess_path, 1950, 2015, target_grid_path, boundary, 15, True, 'final')
                        
                    else:
                        standard_preprocess(raw_path, temp_path, preprocess_path, 2014, 2100, target_grid_path, boundary, 15, True, 'final')






