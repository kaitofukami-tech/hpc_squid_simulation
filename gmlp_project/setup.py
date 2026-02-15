from setuptools import setup, find_packages

setup(
    name="gmlp_project",
    version="0.0.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
)
