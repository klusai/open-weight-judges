from setuptools import setup, find_packages


def read_requirements():
    with open("requirements.txt", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="open-weight-judges",
    version="0.1.0",
    description="Evaluation framework for open-weight, multi-family LLM judge panels",
    author="Mihai Nadăș",
    url="https://github.com/klusai/open-weight-judges",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=read_requirements(),
)
