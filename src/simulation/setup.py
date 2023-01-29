import os
from glob import glob
from setuptools import setup


package_name = 'simulation'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='CARES',
    maintainer_email='cares@auckland.ac.nz',
    description='Gazebo Simulation Package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'environment = simulation.environment:main',
            'training = simulation.training:main'
        ],
    },
)
