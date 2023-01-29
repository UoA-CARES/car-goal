from setuptools import setup

package_name = 'robot_description'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, ['robot_description/world.sdf']),
        ('share/' + package_name, ['robot_description/car.sdf']),
        ('share/' + package_name, ['robot_description/goal.sdf'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='retinfai',
    maintainer_email='retinfai@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
