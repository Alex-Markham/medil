from setuptools import setup


setup(name='medil',
      version='0.5.0',
      packages=['medil'],
      install_requires=['numpy'],
      extras_require={
          'dcor': ['dcor'],
          'GAN' : ['pytorch'],
          'vis' : ['matplotlib', 'networkx']
      }
)
