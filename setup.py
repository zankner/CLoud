from setuptools import setup, find_packages

setup(
    name="cloud",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "llm-foundry>=0.8.0,<0.9.0",
        "openai==1.35.3",
        "datasets==2.19.0",
        "huggingface-hub[hf_transfer]==0.22.2",
        "gradio==4.42.0",
        "vllm==0.5.0.post1",
        "fastapi==0.111.0"
    ],
    author="Zack Ankner",
    author_email="ankner@mit.edu",
    description="Making reward models Critique-out-Loud",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/zankner/CLoud",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.6',
)

