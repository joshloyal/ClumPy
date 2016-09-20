from setuptools import setup


PACKAGES = [
        'clumpy',
        'clumpy.datasets',
        'clumpy.datasets.data',
        'clumpy.tests',
]

def setup_package():
    setup(
        name="ClumPy",
        version='0.1.0',
        description='Implementation of Various Clustering Algorithms',
        author='Joshua D. Loyal',
        url='https://github.com/joshloyal/ClumPy',
        license='MIT',
        install_requires=['numpy', 'scipy', 'scikit-learn', 'prim'],
        packages=PACKAGES,
    )


if __name__ == '__main__':
    setup_package()
