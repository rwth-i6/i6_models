"""
Usage:

Create ~/.pypirc with info:

    [distutils]
    index-servers =
        pypi

    [pypi]
    repository: https://upload.pypi.org/legacy/
    username: ...
    password: ...

(Not needed anymore) Registering the project: python3 setup.py register
New release: python3 setup.py sdist upload

I had some trouble at some point, and this helped:
pip3 install --user twine
python3 setup.py sdist
twine upload dist/*.tar.gz

See also MANIFEST.in for included files.

For debugging this script:

python3 setup.py sdist
pip3 install --user dist/*.tar.gz -v
(Without -v, all stdout/stderr from here will not be shown.)

"""

import os
import shutil
from i6_models.__setup__ import get_version_str, debug_print_file


def main():
    """
    Setup main entry
    """
    # Do not use current time as fallback for the version anymore,
    # as this would result in a version which can be bigger than what we actually have,
    # so this would not be useful at all.
    long_version = get_version_str(verbose=True, fallback="0.0.1+setup-fallback-version", long=True)
    version = long_version[: long_version.index("+")]

    if os.environ.get("DEBUG", "") == "1":
        debug_print_file(".")
        debug_print_file("PKG-INFO")
        debug_print_file("pip-egg-info")
        debug_print_file("pip-egg-info/i6_models.egg-info")
        debug_print_file("pip-egg-info/i6_models.egg-info/SOURCES.txt")  # like MANIFEST

    if os.path.exists("PKG-INFO"):
        if os.path.exists("MANIFEST"):
            print("package_data, found PKG-INFO and MANIFEST")
            package_data = open("MANIFEST").read().splitlines() + ["PKG-INFO"]
        else:
            print("package_data, found PKG-INFO, no MANIFEST, use *")
            # Currently the setup will ignore all other data except in i6_models/.
            # At least make the version available.
            shutil.copy("PKG-INFO", "i6_models/")
            shutil.copy("_setup_info_generated.py", "i6_models/")
            # Just using package_data = ["*"] would only take files from current dir.
            package_data = []
            for root, dirs, files in os.walk("."):
                for file in files:
                    package_data.append(os.path.join(root, file))
    else:
        print("dummy package_data, does not matter, likely you are running sdist")
        with open("_setup_info_generated.py", "w") as f:
            f.write("version = %r\n" % version)
            f.write("long_version = %r\n" % long_version)
        package_data = ["MANIFEST", "_setup_info_generated.py"]

    from setuptools import setup

    authors = [
        dict(name="Christoph Lüscher", email="luescher@cs.rwth-aachen.de"),
        dict(name="Nick Rossenbach", email="rossenbach@cs.rwth-aachen.de"),
        dict(name="Benedikt Hilmes", email="hilmes@cs.rwth-aachen.de"),
        dict(name="Jingjing Xu", email="jxu@cs.rwth-aachen.de"),
        dict(name="Mohammad Zeineldeen", email="zeineldeen@cs.rwth-aachen.de"),
        dict(name="Albert Zeyer", email="zeyer@cs.rwth-aachen.de"),
        dict(name="Peter Vieting", email="vieting@cs.rwth-aachen.de"),
        dict(name="Simon Berger", email="sberger@cs.rwth-aachen.de"),
        dict(name="Eugen Beck", email="ebeck@apptek.com"),
        dict(name="Ping Zheng", email="p.zheng@apptek.com"),
        dict(name="Wilfried Michel", email="wmichel@apptek.com"),
    ]

    setup(
        name="i6_models",
        version=version,
        packages=["i6_models"],
        include_package_data=True,
        package_data={"i6_models": package_data},  # filtered via MANIFEST.in
        description="A collection of PyTorch NN models and parts.",
        author=", ".join(d["name"] for d in authors),
        author_email=", ".join(d["email"] for d in authors),
        url="https://github.com/rwth-i6/i6_models/",
        license="Mozilla Public License 2.0",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        # https://pypi.org/classifiers/
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
            "Operating System :: OS Independent",
        ],
    )


if __name__ == "__main__":
    main()
