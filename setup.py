import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="aries",
    version="0.1.0",
    author="Mike D'Arcy",
    author_email="miked@collaborator.allenai.org",
    description="Code for the ARIES project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache 2.0",
    url="https://github.com/allenai/aries",
    packages=setuptools.find_packages(),
    classifiers=[
        "Topic :: Scientific/Engineering",
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
)

