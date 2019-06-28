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
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
    ],
)
