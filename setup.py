# -*- coding: utf-8 -*-

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gm",
    version="0.0.1",
    author="Daniel Otto",
    author_email="drotto@uw.edu",
    description="Python adaption of the 3-stage glacier model (Roe and Baker, 2014)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
