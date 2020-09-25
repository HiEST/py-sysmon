<h1 align="center">
  py-sysmon
</h1>

  <a href='https://opensource.org/licenses/Apache-2.0'>
    <img src='https://img.shields.io/badge/License-Apache%202.0-blue.svg' alt='License'/>
  </a>

  <a href="https://zenodo.org/badge/latestdoi/267315762">
    <img src="https://zenodo.org/badge/267315762.svg" alt="DOI">
  </a>

</p>

<p align="center">
    <b>py-sysmon</b> python package that allows users to monitor different system's resources usage (cpu, mem, freq, power, etc.).
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a> •
  <a href="#applications">Applications</a> •
  <a href="#installation">Installation</a>
</p>


## Quickstart
Py-SysMon supports different interfaces for gathering system statistics. Check out [py-sysmon's documentation here](https://py-sysmon.readthedocs.io/en/latest). 

### Citation
If you use py-sysmon for your research please cite our [preprint](https://www.arxiv.org/to-be-submitted): 

> Daniel Rivas-Barragan, Francesc Guim-Bernat, Jordà Polo, and David Carrera (2020).
Performance considerations for accelerating video analytics on the edge cloud. *arXiv* 2020.tbd; https://doi.org/tbd

## Applications
Py-SysMon can be applied for different applications:

**Scripts and real examples**: https://github.com/danirivas/py-sysmon/tree/master/examples


## Installation

<p align="center">
  <a href="https://drug2ways.readthedocs.io/en/latest/">
    <img src="http://readthedocs.org/projects/drug2ways/badge/?version=latest"
         alt="Documentation">
  </a>

  <img src='https://img.shields.io/pypi/pyversions/drug2ways.svg' alt='Stable Supported Python Versions'/>
  
  <a href="https://pypi.python.org/pypi/drug2ways">
    <img src="https://img.shields.io/pypi/pyversions/drug2ways.svg"
         alt="PyPi">
  </a>
</p>

The latest stable code can be installed from [PyPI](https://pypi.python.org/pypi/py-sysmon) with:

```python
python -m pip install py-sysmon
```

The most recent code can be installed from the source on [GitHub](https://github.com/danirivas/py-sysmon) with:

```python
python -m pip install git+https://github.com/danirivas/py-sysmon.git
```

For developers, the repository can be cloned from [GitHub](https://github.com/danirivas/py-sysmon) and installed in
editable mode with:

```python
git clone https://github.com/danirivas/py-sysmon.git
cd py-sysmon
python -m pip install -e .
```

## Requirements
```python
numpy==1.19.2
nvidia-ml-py3==7.352.0
pandas==1.1.2
psutil==5.7.2
-e git+https://github.com/wkatsak/py-rapl.git@194fabad1144cad7edb4fdb1c8e17edb57deb8b1#egg=py_rapl
tqdm==4.49.0
```

