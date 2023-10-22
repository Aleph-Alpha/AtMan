import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="atman-open-images-eval",
    version="0.0.1",
    author="Aleph-Alpha",
    author_email="",
    description= "open images dataset eval code for explainability benchmark",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    install_requires= required,
    python_requires='>=3.6',
    include_package_data=True,
    keywords=[
        "PyTorch",
        "machine learning",
        ],
    classifiers=[
        "Intended Audience :: Science/Research",
    ],
    test_suite='nose.collector',
    tests_require=['nose']
)
