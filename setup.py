from typing import List

from setuptools import find_packages, setup


def get_requirements(file_path: str) -> List[str]:
    """Get the requirements/dependencies (packages) in a list."""
    with open(file_path) as f:
        lines = f.readlines()
        requirements = [line.rstrip("\n") for line in lines]

        return requirements


setup(
    name="defectDetect",
    version="1.0",
    author="ManishW",
    author_email="manishdrw1@gmail.com",
    description="",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
