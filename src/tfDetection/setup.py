#!/usr/bin/env python
"""
Simple check list to create the package for pypi.
1. Change the version in __init__.py, setup.py as well as docs/source/conf.py.
2. Unpin specific versions from setup.py (like isort).
2. Commit these changes with the message: "Release: VERSION"
3. Add a tag in git to mark the release: git tag VERSION -m'Adds tag VERSION for pypi"
   Push the tag to git: git push --tags origin master
4. Build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).
   For the wheel, run: "python setup.py bdist_wheel" in the top level directory.
   (this will build a wheel for the python version you use to build it).
   For the sources, run: "python setup.py sdist"
   You should now have a /dist directory with both .whl and .tar.gz source versions.
5. Check that everything looks correct by uploading the package to the pypi test server:
   twine upload dist/* -r pypitest
   (pypi suggest using twine as other methods upload files via plaintext.)
   You may have to specify the repository url, use the following command then:
   twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/
   Check that you can install it in a virtualenv by running:
   pip install -i https://testpypi.python.org/pypi transformers
6. Upload the final version to actual pypi:
   twine upload dist/* -r pypi
7. Copy the release notes from RELEASE.md to the tag in github once everything is
looking hunky-dory.
8. Update the documentation commit in .circleci/deploy.sh for the accurate
   documentation to be displayed
9. Update README.md to redirect to correct documentation.
"""

# Custom Packages
# ======================================================================================
from setuptools import find_packages
from setuptools import setup


setup(
    name="tfDetection",
    version="0.1.0",
    author="Trieu Phat Luu",
    author_email="tpluu2207@gmail.com",
    description="",
    keywords="Object Detection, tensorflow",
    license="Apache",
    url="https://github.com/trieuphatluu/",
    packages=find_packages(),
    install_requires=[],
    python_requires=">=3.6.0",
    include_package_data=True,
    test_suite="tests",
    zip_safe=False,
    classifiers=[
        "Development Status :: 2 - Pre-alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
