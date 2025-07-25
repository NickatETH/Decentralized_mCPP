from setuptools import find_packages, setup

package_name = 'drone'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'shapely', 'numpy'],
    zip_safe=True,
    maintainer='nick',
    maintainer_email='ntruttmann@ethz.ch',
    description='ROS2 UAV agent node package',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'uav_agent = drone.uav_agent:main',
        ],
    },
)
