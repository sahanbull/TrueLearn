from setuptools import setup

setup(
    name='TrueLearn',
    version='1.0',
    packages=['truelearn_experiments'],
    url='',
    license='',
    author='',
    author_email='',
    description='TrueLearn: A Bayesian algorithm that models background knowledge and novelty of lifelearn learners to predict engagement',
    install_requires=[
        'numpy>=1.14.1',
        'pandas>=0.22.0',
        'scipy>=1.0.1',
        'nltk>=3.2.5',
        'xmltodict>=0.11.0',
        'ujson>=1.35',
        'scikit-learn>=0.19.1',
        'pyspark==2.4.3',
        'trueskill==0.4.5',
        'mpmath==1.1.0']
)