from setuptools import setup

package_name = 'final_race_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
    'setuptools',
    'PyQt5',
    'matplotlib',
    'numpy',
    'scipy'
    ],
    zip_safe=True,
    maintainer='lucien',
    maintainer_email='2559346258@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'main_agent = final_race_pkg.main_agent:main',
        'opp_agent = final_race_pkg.opp_agent:main',
        'traj_killer = final_race_pkg.traj_killer:main',
    ],
    },
)
