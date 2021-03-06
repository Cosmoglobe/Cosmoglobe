import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cosmoglobe",
    version="0.1.8",
    author="Metin San",
    author_email="metinisan@gmail.com",
    description="The Cosmoglobe Sky Model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Cosmoglobe/Cosmoglobe",
    packages=setuptools.find_packages(),
    package_data={"": ["*.dat","BP7_70GHz_nocmb_n0256.fits"]}, 
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=[
        "healpy",
        "astropy",
        "numpy",
        "h5py>=3.0",
        "scipy",
        "numba",
    ]
)