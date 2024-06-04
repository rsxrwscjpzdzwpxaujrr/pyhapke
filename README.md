# pyhapke

Forked from [pyhapke](https://github.com/jordankando/pyhapke) by Jordan Ando.

## Description

`pyhapke` is an implementation of Hapke's radiative transfer model (Hapke 1981), which is commonly used to model planetary surfaces. This model can be used to relate reflectance to single-scattering albedo of various materials.

This package is built around the HapkeRTM class, which takes input parameters of incidence, emmittance, and phase angles, as well as phase, and can be used to compute either single scattering albedo or reflectance, given the other. Included constants consist primarily of relevant physical properties of lunar regolith and ice, as well as standard viewing geometries. Each verson of Hapke's model, for bidirectional reflectance, reflectance factor, and radiance factor, is included.

To install, clone or download this repo. From the top-level folder (the one containing `pyproject.toml`), run `pip install -e .` to install the editable version of the package  (`pyhapke` is not yet available on the PyPI for regular pip installation).

To start using the pyhapke package, simply:

```python
from pyhapke import HapkeRTM
model = HapkeRTM()
refl = model.hapke_function(ssa=0.12)
```

pyhapke is designed to run with numpy and scipy as its primary dependencies, which are commonly found in other scientific Python applications, and can be easily installed through Anaconda or pip.

## License

This software is licensed under the [MIT license](./LICENSE).
