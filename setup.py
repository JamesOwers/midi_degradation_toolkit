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
    keywords='MIDI ACME melody music dataset',
    url="https://github.com/JamesOwers/midi_degradation_toolkit",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'tqdm',
        'mir_eval',
        'pretty_midi',
        'torch',
        'seaborn'
    ],
    python_requires='~=3.7',
    classifiers=[
        "Development Status :: 1 - Planning",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Sound/Audio :: MIDI"
    ],
    dependency_links=['https://github.com/craffel/pretty-midi/tarball/'
                      'master#egg=0.2.8'],
    zip_safe=False,
    setup_requires=["pytest-runner"],
    tests_require=["pytest"]
)
