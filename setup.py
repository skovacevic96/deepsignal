from setuptools import find_packages, setup
setup(
    name='deepsignal',
    packages=find_packages(),
    version='0.1.0',
    description='Digital signal processing utils mash',
    author='University of Montenegro',
    license='MIT',
	install_requires=['numpy==1.19.5','scipy==1.4.1','tensorflow==2.4.1','librosa==0.7.2'],
)