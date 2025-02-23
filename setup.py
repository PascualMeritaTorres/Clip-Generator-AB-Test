from setuptools import setup, find_packages

setup(
    name="clip_generator",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "fal-client",
        "python-dotenv",
        "moviepy",
    ],
) 