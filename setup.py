from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='smcper',
    version='0.1.2',    
    description='A python package that allows you to import pytorch layers form model '
    'compression using the scalable model compression by entropy penalized reparameterization'
    'paper',
    url='https://github.com/Dan8991/SMCEPR_pytorch',
    author='Daniele Mari',
    author_email='daniele.mari@phd.unipd.it',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['smcper'],
    install_requires=['torch',
                      'torchvision',
                      'compressai'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.10',
    ],
)
