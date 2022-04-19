import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    reqs = []
    for line in f:
        line = line.strip()
        reqs.append(line.split('==')[0])

setuptools.setup(
    name="ct_loss",
    version="0.0.1",
    author="Shaojie Jiang",
    author_email="shaojiejiang.1991@gmail.com",
    description="The contrastive token loss for reducing generative repetition of augoregressive neural language models.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ShaojieJiang/CT-Loss",
    project_urls={
        "Bug Tracker": "https://github.com/ShaojieJiang/CT-Loss/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=reqs,
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
)
