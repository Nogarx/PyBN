import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pybn",
    version="1.0.0",
    author="Mario Franco",
    author_email="mfrancom88@gmail.com",
    description="A package for Boolean Network research.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Nogarx/PyBN",
    project_urls={
        "Bug Tracker": "https://github.com/Nogarx/PyBN/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.19.5", 
        "ray>=1.0.1", 
        "ray[default]>=1.0.1", 
        "tqdm>=4.60.0", 
        "matplotlib>=3.3.4", 
        "filelock>=3.0.12"]
)