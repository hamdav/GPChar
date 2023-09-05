from setuptools import setup, find_packages

setup(
        name="gpchar",
        version="0.1.0",
        description="Characterizing functions with Gaussian Processes",
        author="David Hambraeus",
        package_dir={"": "src"},
        packages=find_packages(where="src"),
        install_requires=["scikit-learn", "numpy", "scipy", "dash", "plotly", "pandas"],
        )