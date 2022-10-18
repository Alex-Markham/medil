from setuptools import setup


def long_description():
    with open("README.md") as readme:
        return readme.read()


setup(
    name="medil",
    version="0.7.0",
    author="Alex Markham",
    author_email="alex.markham@causal.dev",
    description="This package is for causal modeling, originally focusing on the measurement dependence inducing latent (MeDIL) causal model framework, but now including more general methods for causal discovery and inference.",
    long_description_content_type="text/markdown",
    long_description=long_description(),
    license="Cooperative Non-Violent Public License v7 or later (CNPLv7+)",
    packages=["medil"],
    install_requires=["numpy"],
    extras_require={
        "dcor": ["dcor"],
        "GAN": ["pytorch-lightning"],
        "vis": ["matplotlib", "networkx"],
        "all": ["dcor", "torch", "matplotlib", "networkx"],
    },
)
