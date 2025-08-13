from setuptools import find_packages, setup

setup(
    name="Medical-Chatbot",
    version="0.1.0",
    author="Manail Fatima",
    author_email="manailfatima2@gmail.com",
    
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[]
)