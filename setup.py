from setuptools import setup, find_packages

setup(
    name='dual_fair_boost',
    version='0.1.0',
    description='A Fairness Classication model based in Gradient Boosting',
    author='Giovani Valdrighi',
    author_email='giovani.valdrighi@gmail.com',
    packages=find_packages(),
    install_requires=[
        # List any dependencies your package requires
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
