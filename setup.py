from setuptools import setup
setup(
    name='sophius',
    version='0.1.dev',
    packages=['sophius'],
    install_requires=[
        'numpy >= 1.12',
        'torch >= 1.4.0',
        'torchvision >= 0.5.0',
        'bitmath >= 1.3.0',
        'mathplotlib >= 3.2.0'
        ]
)