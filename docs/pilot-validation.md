# Pilot validation

Use the synthetic recordings first. Their [manifest](../fixtures/synthetic/expected.json) specifies the arrivals, source hashes and reference fits. Record the SeisSlate commit, Python and dependency versions, operating system, run date, sample count, and the outcome of each check.

## Synthetic review

1. Select **Synthetic example**. Confirm that the original receiver has a recording gap at B07 and that its calibration arrival is still visible. Confirm that the all-pair fit uses seven blows and the strict imported fit uses B01, B02, B03 and B08.
2. Select B01 and **Original**, then use **+1 sample**. Cancel the preview and confirm that the applied lag stays at −40 ms. Repeat and select **Apply**. Check that the lag is −39 ms, the calibration pick and imported reference fits are unchanged, and the reviewed slope is approximately 507.67918 m/s.
3. Double-click the original receiver's waveform, then drag across a short interval that covers the revised B01 arrival. Select **Mask interval** and **Apply**. Check that the calibration pick remains, the lag becomes unavailable, and requested fit inclusion is retained. Mark the interval again, select **Restore interval**, and apply. Check that the lag returns. Use **Toggle fit inclusion** separately and check that its lag stays visible.
4. Select **Save review…** and **Export CSV…**. Quit, launch a fresh process, and reopen the review. Compare the picks, masks, fit membership and coefficients with the prior export. Undo and redo the last edit. Check that a moved source with matching bytes can be relinked and that a different recording is rejected.

## Windows workstation

Install the [desktop pilot](../README.md#install-the-desktop-pilot) from the checkout. Use x64 CPython 3.13. If the guest is Windows 11 Arm, select the x64 Python installer and check that the environment reports `win-amd64`. This exercises the packaged x64 dependencies through Windows emulation. Record VM results separately from results on the intended workstation. [Windows on Arm emulation](https://learn.microsoft.com/en-us/windows/arm/apps-on-arm-x86-emulation) · [Pyrocko release wheels](https://pypi.org/project/pyrocko/2026.11.5/#files)

```powershell
& .\.venv\Scripts\python.exe -c "import sysconfig; print(sysconfig.get_platform())"
& .\.venv\Scripts\python.exe -B tools\check_pilot.py
```

The runner checks the synthetic fixture, the application tests including Qt interactions, and the Snuffler sample-nudge probe. Use the default Qt backend on Windows. The runner reports the actual backend, so you can label a headless run correctly. It uses temporary Pyrocko and Matplotlib settings. Snuffler still uses its standard user plugin directory. Record the runner output, then open `.venv\Scripts\seisslate.exe` from File Explorer and complete the [synthetic review](#synthetic-review).

Run the synthetic review on the intended Windows workstation. Record the installation method, whether administrator rights were needed during setup, and how the analyst launched the workbench from the desktop. Check native file dialogs, resize and 100% and 150% scaling. Repeat the save and reopen in a folder whose name contains spaces and non-ASCII characters. For a recovery check, work on a disposable copy and verify that a failed save leaves a complete readable review. Record the time from applying an edit to seeing the consistent waveform, table and fit. Include the data size and machine details with latency results.

### Snuffler interaction probe

From the checkout on Windows x64, set up the isolated probe with Python 3.13.

```powershell
py -3.13 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --only-binary=:all: "pyrocko==2026.11.5" "PyQt5==5.15.11"
& .\.venv\Scripts\python.exe -B tools\snuffler_probe.py --check
```

Run again without `--check` to exercise the visible window. Select B01's earlier Original pick at 00:00:00.360 UTC in the Markers table. Click **Nudge +1 sample** and verify that the pick moves to 00:00:00.361 UTC and the calibration marker stays at 00:00:00.400 UTC. Snuffler may create its standard user plugin directory at first launch. Pyrocko provides a wheel for Windows x64 and CPython 3.13. [Pyrocko 2026.11.5 files](https://pypi.org/project/pyrocko/2026.11.5/#files)

The automated interaction passed on Apple Silicon macOS with offscreen Qt. The native Windows run is pending.

## Independent analyst and field cases

Have a second analyst repeat the synthetic review without the developer operating the interface. Ask the analyst to explain the changed and unchanged observations and reopen the saved review independently. Compare completion time and errors with the existing viewer and notebook workflow using the same set of review decisions.

Only use approved field recordings. Record the embedded channel IDs, receiver roles, clock-alignment evidence, geometry sources and units, and the evidence that each pair is from the same blow. Include one clear group and one ambiguous case. Record numerical reproduction, Windows usability, and the basis for any field interpretation as separate outcomes.
