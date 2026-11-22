# SeisSlate

SeisSlate is a Windows-first project for reviewing seismic waveforms and seeing how edits affect the analysis. Its first use case is comparing hammer strikes recorded by two seismic nodes. The planned workbench will let analysts move an arrival pick, exclude or restore part of a waveform, and recalculate the fit while preserving the original MiniSEED recordings.

## Current status

The repository contains a synthetic two-node fixture and a reproducible comparison of two fits. The waveform editor and Windows workflow are still under development.

The fixture includes closely spaced strikes, a deliberately mistimed pick, and an arrival missing because of a recording gap. The [baseline table](fixtures/synthetic/baseline.csv) shows the picks, signed arrival delays, distance differences, and residuals. The [fixture manifest](fixtures/synthetic/expected.json) records the source hashes and expected fits. The exercise uses synthetic geometry and does not establish a field velocity.

## Run the fixture checks

Use an activated Python environment. The fixture checks use ObsPy 1.5.2 and NumPy 2.5.0.

```sh
python -m pip install "obspy==1.5.2" "numpy==2.5.0"
python -B -m unittest discover -s fixtures -p "test_*.py"
```

The tests regenerate the fixture in a temporary directory, compare it with the checked-in examples, and check the signed delays and regression outputs. To generate your own copy, give the script a new output directory.

```sh
python -B fixtures/generate_synthetic.py outputs/synthetic-example
```

## Related work

The interface design draws on waveform navigation and picking in Pyrocko Snuffler, PyRefra, and Refrapy. [Snuffler](https://pyrocko.org/docs/current/apps/snuffler/manual.html) · [PyRefra](https://github.com/HZeyen/PyRefra) · [Refrapy](https://github.com/viictorjs/Refrapy)
