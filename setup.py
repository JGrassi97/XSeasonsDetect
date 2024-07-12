from setuptools import setup, find_packages

setup(
    name='RCC',
    version='0.1.0',
    author='Jacopo Grassi',
    author_email='jacopo.grassi@polito.it',
    description='Una breve descrizione del progetto',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    # url='https://github.com/tuo-username/nome_progetto',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
        'numpy>=1.18.0',
        'pandas>=1.0.0',
        'scikit-learn>=0.22.0',
        'matplotlib>=3.1.0',
        'seaborn>=0.10.0',
    ],
    extras_require={
        'dev': [
            'pytest>=5.3.5',
            'jupyterlab>=2.0.0',
            'flake8',
            'black',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.1',
)
