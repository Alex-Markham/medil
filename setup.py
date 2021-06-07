from setuptools import setup


def long_description():
    with open("README.md") as readme:
        return readme.read()


setup(
    name="medil",
    version="0.6.0",
    author="Alex Markham",
    author_email="alex.markham@causal.dev",
    description="This package is for causal inference, focusing on the Measurement Dependence Inducing Latent (MeDIL) Causal Model framework.",
    long_description_content_type="text/markdown",
    long_description=long_description(),
    license="Cooperative Non-Violent Public License v6 or later (CNPLv6+)",
    packages=["medil"],
    install_requires=["numpy"],
    extras_require={
        "dcor": ["dcor"],
        "GAN": ["pytorch-lightning"],
        "vis": ["matplotlib", "networkx"],
        "all": ["dcor", "torch", "matplotlib", "networkx"],
    },
)
