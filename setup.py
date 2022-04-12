import io
import setuptools

# Package meta-data.
NAME = 'Lessen Supervisison'
DESCRIPTION = 'Semantic Segmentation unsupervised and weakly supervised techniques.'
URL = 'https://github.com/martafdezmAM/lessen_supervision/'
EMAIL = 'martafdezm95@gmsil.com'
AUTHOR = 'Marta Fernández Moreno'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '1.0.0'


setuptools.setup(
    name=NAME,
    packages=['seg_kanezaki', 'pixelpick'],
    package_dir={'':'models'},
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    #packages=setuptools.find_packages(exclude=['tests', 'docs', '.git', '.idea',]),
    classifiers=(
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools'
        'Topic :: Scientific/Engineering :: Deep Learning',
    ),
)
