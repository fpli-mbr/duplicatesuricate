from setuptools import setup

setup(name='duplicatesuricate',
      version='0.3.4',
      description='Entity resolution algorithm implemented with scikit-learn',
      author='Amber Ocelot',
      author_email='AmberOcelot@gmail.com',
      license='na',
      packages=['duplicatesuricate'],
      zip_safe=False,
      url='https://github.com/ogierpaul/duplicatesuricate',
      download_url='https://github.com/ogierpaul/duplicatesuricate.git',
      keywords=['pandas', 'data cleaning',
                'entity matching', 'entity resolution', 'record linkage'],
      classifiers=[
          'Programming Language :: Python :: 3',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Intended Audience :: Developers',
          'Topic :: Utilities'
      ],
      install_requires=['neatmartinet', 'pandas', 'numpy', 'fuzzywuzzy','scikit-learn']
      )
