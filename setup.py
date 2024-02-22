from setuptools import setup


def long_description():
    with open("README.md") as readme:
        return readme.read()


setup(
    name="medil",
    version="0.7.0",
    author="Alex Markham",
    author_email="alex.markham@causal.dev",
    description="MeDIL is a Python package for causal factor analysis with the (me)asurement (d)ependence (i)nducing (l)atent causal model framework.",
    long_description_content_type="text/markdown",
    long_description=long_description(),
    license="GNU Affero General Public License version 3 or later (AGPLv3+)",
    packages=["medil"],
    install_requires=["numpy"],
    extras_require={
        "vis": ["matplotlib", "networkx"],
        "dgm": ["dcor", "xicorrelation", "torch"],
        "all": ["matplotlib", "networkx", "dcor", "dcor", "xicorrelation", "torch"],
    },
)
