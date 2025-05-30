from setuptools import setup, find_packages

setup(
    name='inference',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,  
    install_requires=[
        'click',
        'torch',
        'scikit-learn',
    ],
    entry_points={
        'console_scripts': [
            'inference=inference.cli:main',
        ],
    },
)
