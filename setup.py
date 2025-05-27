from setuptools import setup, find_packages

setup(
    name='inference',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'pandas',
        'scikit-learn',
    ],
    entry_points={
        'console_scripts': [
            'inference=inference.cli:main',
        ],
    },
)
