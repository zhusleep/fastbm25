import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fastbm25",
    version="0.0.2",
    author="zhusleep",
    author_email="zhuflower@qq.com",
    description="Fast text match algorithm implementation for bm25",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhusleep/fastbm25",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)

