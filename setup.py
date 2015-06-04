from setuptools import setup, find_packages

def readme():
    with open('README.rst') as f:
        return f.read()

# Hackishly inject a constant into builtins to enable importing of the
# package before the library is built.
import sys
if sys.version_info[0] < 3:
    import __builtin__ as builtins
else:
    import builtins
builtins.__ROTATION_SETUP__ = True
import rotation


setup(name = "rotation",
    version = rotation.__version__,
    description = "Estimating rotation periods from time-series photometry",
    long_description = readme(),
    author = "Timothy D. Morton",
    author_email = "tim.morton@gmail.com",
    url = "https://github.com/timothydmorton/kepler-rotation",
    packages = find_packages(),
    #package_data = {'rotation':['data/*']},
      scripts = ['scripts/prepare_photometry',
                 'scripts/prepare_aigrain'],
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research',
      'Operating System :: OS Independent',
      'Programming Language :: Python',
      'Topic :: Scientific/Engineering :: Astronomy'
      ],
    install_requires=['acor>=1.1.1'],
    zip_safe=False
) 
