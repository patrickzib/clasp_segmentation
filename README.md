# ClaSP - Time Series Segmentation


## Installation

You can also install  the project from source.

## Build from Source

First, download the repository.
```
git clone https://github.com/patrickzib/clasp_segmentation.git
```

Change into the directory and build the package from source.
```
pip install .
```


## Example Code

```python

from clasp.annotation.clasp import ClaSPSegmentation, find_dominant_window_sizes
X = ... # load dataset
dominant_period_size = find_dominant_window_sizes(X)
clasp = ClaSPSegmentation(dominant_period_size, n_cps=1)
found_cps = clasp.fit_predict(X)
profiles = clasp.profiles
scores = clasp.scores
```

See also the notebooks-folder for a Jupyter-Notebook

## Citing

If you use this algorithm or publication, please cite:

```bibtex
@inproceedings{clasp2021,
  title={ClaSP - Time Series Segmentation},
  author={Sch"afer, Patrick and Ermshaus, Arik and Leser, Ulf},
  booktitle={CIKM},
  year={2021}
}
```
