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
builtins.__KEPROT_SETUP__ = True
import keprot


setup(name = "keprot",
    version = keprot.__version__,
    description = "Estimating rotation periods from Kepler photometry",
    long_description = readme(),
    author = "Timothy D. Morton",
    author_email = "tim.morton@gmail.com",
    url = "https://github.com/timothydmorton/kepler-rotation",
    packages = find_packages(),
    #package_data = {'':['data/*']},
    scripts = ['scripts/prepare_photometry'],
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
