<h1 align="center">
  py-sysmon
</h1>

  <a href='https://opensource.org/licenses/Apache-2.0'>
    <img src='https://img.shields.io/badge/License-Apache%202.0-blue.svg' alt='License'/>
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
Performance Characterization of Video Analytics Workloads in Heterogenous Edge Infrastructures. *arXiv* 2020.tbd; https://doi.org/tbd

## Applications
Py-SysMon can be applied for different applications:

**Scripts with examples**: https://github.com/HiEST/py-sysmon/tree/master/scripts

**Docker files**: https://github.com/HiEST/py-sysmon/tree/master/docker

## Installation

The most recent code can be installed from the source on [GitHub](https://github.com/HiEST/py-sysmon) with:

```python
python -m pip install git+https://github.com/HiEST/py-sysmon.git
```

For developers, the repository can be cloned from [GitHub](https://github.com/HiEST/py-sysmon) and installed in
editable mode with:

```python
git clone https://github.com/HiEST/py-sysmon.git
cd py-sysmon
python -m pip install -e .
```

## Requirements
```python
numpy==1.19.2
pandas==1.1.2
psutil==5.7.2
-e git+https://github.com/wkatsak/py-rapl.git@194fabad1144cad7edb4fdb1c8e17edb57deb8b1#egg=py_rapl
tqdm==4.49.0
```

