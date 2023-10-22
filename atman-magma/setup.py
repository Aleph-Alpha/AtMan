import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="atman-magma",
    version="0.0.1",
    author="Aleph-Alpha",
    author_email="",
    description= "open source impl of atman on magma",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Aleph-Alpha/AtMan",
    packages=setuptools.find_packages(),
    install_requires= None,
    #python_requires='>=3.6',
    include_package_data=True,

    classifiers=[

        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",

    ],
)
