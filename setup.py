from setuptools import setup, find_packages

setup(
    name='XSeasonDetect',
    version='0.1',
    description='A brief description of your project',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Jacopo Grassi',
    author_email='jacopo.grassi@polito.it',
    packages=find_packages(where='src'),  # Automatically find packages in 'src'
    package_dir={'': 'src'},  # Indicate that packages are under the 'src' directory
    include_package_data=True,
    install_requires=[
        # Add your package dependencies here
    ],
)