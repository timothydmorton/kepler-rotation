__version__ = '0.1'

try:
    __KEPROT_SETUP__
except NameError:
    __KEPROT_SETUP__ = False

if not __KEPROT_SETUP__:
    __all__ = []

