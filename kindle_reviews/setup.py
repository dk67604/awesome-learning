from setuptools import setup, find_packages

setup(
    name='sentimental_analysis',
    packages=find_packages(where="src", include=['kindle_reviews.libs*']),
    package_dir={"": "src"},
    version='0.1.0',
    description='Sentimental Analysis',
    author='',
    license='MIT',
)
