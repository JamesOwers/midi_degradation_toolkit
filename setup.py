"""Script for setuptools"""
import setuptools

with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

setuptools.setup(
    name="mdtk",
    version="2019.06rc1",
    author="James Owers",
    author_email="james.f.owers@gmail.com",
    description="A toolkit for creating datasets of Altered and Corrupted "
                "MIDI Excerpts (ACME datasets)",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/JamesOwers/midi_degradation_toolkit",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'pretty_midi'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
    ],
    dependency_links=['https://github.com/craffel/pretty-midi/tarball/'
                      'master#egg=0.2.8'],
    zip_safe=False
)
