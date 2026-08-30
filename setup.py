from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension("colibri_runtime._rowio", ["native/rowio.c"]),
    ],
)
