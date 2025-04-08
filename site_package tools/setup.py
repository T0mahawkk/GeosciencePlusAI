from setuptools import setup, find_packages

setup(
    name="GeosciencePlusAI",
    version="0.1",
    #packages=find_packages(),
    packages=[r'C:\Users\aa2142\Git\Projects\EAGE_2025\GeosciencePlusAI\Lib\ClassificationFunctions.py', r'C:\Users\aa2142\Git\Projects\EAGE_2025\GeosciencePlusAI\Lib\ClusteringFunctions.py', r'C:\Users\aa2142\Git\Projects\EAGE_2025\GeosciencePlusAI\Lib\ClassificationFunctions\DataVisualisation.py'],  # Include the Lib package
    #packages=["ClassificationFunctions", "ClusteringFunctions", "DataVisualisation"],  # Explicitly include the Lib package
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",

    ],
    description="A package for geoscience data visualization, clustering, and classification.",
    author="Farah Rabie",
    author_email="fr2007@hw.ac.uk",
    url="https://github.com/T0mahawkk/GeosciencePlusAI.git",  # Replace with your repository URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)