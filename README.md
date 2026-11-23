# SeisSlate

SeisSlate is a local desktop workbench for reviewing first arrivals from hammer strikes recorded by two seismic nodes. Move a pick, exclude or restore a waveform interval, or revise which blows contribute to the fit. Save the review so a colleague can reproduce the result from the original MiniSEED recordings.

## Current status

The pilot uses Pyrocko Snuffler for waveform navigation and selection. The checked-in example uses synthetic recordings. Windows workstation testing and field validation are pending. [Pilot validation procedure](docs/pilot-validation.md)

The example includes closely spaced strikes, a deliberately mistimed pick, and an arrival missing because of a recording gap. The fit relates signed distance difference in metres to signed arrival delay in seconds, both calculated as original minus calibration. The two imported reference fits use all available pairs and the strict interval `−0.06 < delay < 0.08` seconds. Subsequent review edits leave that baseline fixed. The displayed slope describes the numerical fit. Interpreting field velocity requires evidence about blow pairing, clocks and geometry. [Baseline table](fixtures/synthetic/baseline.csv) · [Fixture manifest](fixtures/synthetic/expected.json)

## Install the desktop pilot

From the checkout on Windows x64 with Python 3.13:

```powershell
py -3.13 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -e ".[desktop]"
```

After setup, open `.venv\Scripts\seisslate.exe` from File Explorer. Select **Synthetic example** to load the checked-in exercise, **Open pair…** to select two MiniSEED files, or **Open review…** to resume a session. The viewer requires distinct trace IDs and nonoverlapping segments within each receiver track. It reads MiniSEED 2 waveform records and preserves gaps. See the [pilot validation procedure](docs/pilot-validation.md) for the edit sequence and Windows checks.

On macOS, install the same desktop extra in a virtual environment and launch the example from the checkout.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[desktop]'
python -m seisslate.ui --example
```

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

Run the core and desktop tests from the checkout after installing the desktop extra.

```sh
python -B -m unittest discover -s tests
```

## Related work

The interface design draws on waveform navigation and picking in Pyrocko Snuffler, PyRefra, and Refrapy. [Snuffler](https://pyrocko.org/docs/current/apps/snuffler/manual.html) · [PyRefra](https://github.com/HZeyen/PyRefra) · [Refrapy](https://github.com/viictorjs/Refrapy)
