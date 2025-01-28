# <span style="font-size:larger;">tRP</span>

tRP (**Ruddlesden-Popper Tolerance Factor**) is a descriptor designed to predict stable RP phases. 

![tRP Banner Image](https://github.com/CNMD-POSTECH/tRP/blob/main/Figure/tRP.png?raw=true)

---

## <span style="font-size:larger;">Installation and Requirements</span>

### <span style="font-size:larger;">Install from PyPI</span>

This is the recommended method for installing tRP:

```bash
git clone https://github.com/CNMD-POSTECH/tRP
cd tRP
```

```bash
conda env create -f setup.yaml
conda activate tRP_env
pip install --upgrade pip
pip install -e .
```

---

## <span style="font-size:larger;">Usage</span>

Before running the commands, make sure to set the `PYTHONPATH` environment variable:

```bash
export PYTHONPATH=$(pwd)
```

### <span style="font-size:larger;">Predicton RP Phase</span>

To predict stable RP phase, use the following command:

```bash
rp-prediction
    --config=./SISSO/src/config_prediction.yaml
```

### <span style="font-size:larger;">Feature Extraction</span>

To extract important features, use the following command:

```bash
extract-feature 
    --config=./SHAP/src/config_shap.yaml
```

### <span style="font-size:larger;">Descriptor Extraction</span>

To extract descriptors, use the following command:

```bash
extract-descriptor 
    --config=./SISSO/src/config_extraction.yaml
```

### <span style="font-size:larger;">Criterion Extraction</span>

To extract criteria, use the following command:

```bash
extract-descriptor
    --config=./SISSO/src/config_criterion.yaml
```

---

## <span style="font-size:larger;">References</span>

If you use tRP in your research, please consider citing the following work:

- Hyo Gyeong Shin, Eun Ho Kim, Jaeseon Kim, Hyo Kim, Donghwa Lee.  
  Ruddlesden-Popper Tolerance Factor: An Indicator Predicting Stability of 2D Ruddlesden-Popper Phases.

---

## <span style="font-size:larger;">Contact</span>

For inquiries, feel free to reach out to:
- Hyo Gyeong Shin ([hyogyeong@postech.ac.kr](mailto:hyogyeong@postech.ac.kr))

For bug reports or feature requests, please open an issue on [GitHub Issues](https://github.com/CNMD-POSTECH/tRP/issues).

---

## <span style="font-size:larger;">License</span>

tRP is distributed under the [MIT License](MIT.md).

---

## <span style="font-size:larger;">Contributors</span>

This repository includes contributions from Hyo Gyeong Shin, Donghwa Lee, and other collaborators.

<a href="https://github.com/CNMD-POSTECH/tRP/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=CNMD-POSTECH/tRP" />
</a>