import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="BuilT",
    version="0.0.1",
    author="JongYoon Lim",
    author_email="jy.lim@auckland.ac.nz",
    description="Easily build your trainer for DNNs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UoA-CARES/BuilT",
    project_urls={
        "Bug Tracker": "https://github.com/UoA-CARES/BuilT/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
)