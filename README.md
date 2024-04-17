# TSFUNES

This repository holds the code for time series analysis with GAN, supervised learning etc.

## Install


### Package static install

Build package and install wheel file
```bash
    python -m build
    pip install dist/tsfunes-0.0.1-py3-none-any.whl
```

### Package dynamic install (development mode)

Use development mode in order to have the package path import:

```bash
    pip install --editable .
```

### TimeGAN

- Vanilla GAN
- Wasserstein GAN
- Wasserstein GAN with gradient penalty


## File Description
./src/datautils/data_scrap.py -------------- data scrap including fetch, filtering and processing

./src/tgan/inference.py -------------- data scraping, inference and training

./src/tgan/daily_inference.py -------- daily scrap-inference process, can be set as routine
