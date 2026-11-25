# Pilot validation

Use the synthetic recordings first. Their [manifest](../fixtures/synthetic/expected.json) specifies the arrivals, source hashes and reference fits. Record the SeisSlate commit, Python and dependency versions, operating system, run date, sample count, and the outcome of each check.

## Synthetic review

1. Select **Synthetic example**. Confirm that the original receiver has a recording gap at B07 and that its calibration arrival is still visible. Confirm that the all-pair fit uses seven blows and the strict imported fit uses B01, B02, B03 and B08.
2. Select B01 and **Original**, then use **+1 sample**. Cancel the preview and confirm that the applied lag stays at −40 ms. Repeat and select **Apply**. Check that the lag is −39 ms, the calibration pick and imported reference fits are unchanged, and the reviewed slope is approximately 507.67918 m/s.
3. Double-click the original receiver's waveform, then drag across a short interval that covers the revised B01 arrival. Enter a reason, then select **Mask interval** and **Apply**. Check that the calibration pick remains, the lag becomes unavailable, and requested fit inclusion is retained. Mark the interval again, enter a reason, select **Restore interval**, and apply. Check that the lag returns. Enter a reason and use **Toggle fit inclusion** separately and check that its lag stays visible.
4. Select **Save review…** and **Export CSV…**. Quit, launch a fresh process, and reopen the review. Compare the picks, masks, fit membership and coefficients with the prior export. Undo and redo the last edit. Check that a moved source with matching bytes can be relinked and that a different recording is rejected.

## Raw pick evidence

1. Select B01 and **Original**, then select **Review now**. The raw dock identifies the selected blow and receiver. If the selection changes before you confirm, open the dock again for the new target. Use the sample control to read the original sample index, UTC, raw value and dtype. Snuffler can still filter its navigation view. The raw dock always reads the saved recording samples.
2. Edit the first sample and exclusive stop to inspect a different span. For onset bounds, enter both inclusive sample indices. Equal endpoints state a zero-width bracket. Select **Show raw span** after editing those fields. Choose **Accepted** or **Unresolved**, enter a reviewer label and reason, and select **Confirm review**. Select **Apply** to keep the decision and the confirmed raw evidence in one revision. **Cancel preview** discards the draft.
3. Mask and restore an interval that intersects the captured span. Reopen the raw review and check that the historical decision is still visible. The attestation is stale after both edits, even when replay finds the original data and restores the view. Confirm and apply a new review to renew the attestation. Test **Undo** and **Redo** with the evidence dock open.
4. Save and reopen the review. Check that the source, point, bounds and recorded raw span agree. Keep the review file and its referenced recordings. Compare the fit export with **File → Export evidence CSV…**, which includes the applied per-receiver decisions and replay status. File export checks recorded settings and sample identity. View restoration is checked in the desktop window.

For a missing arrival such as B07 Original, choose **Mark unresolved**. Enter the intended UTC and a reason, select **Show raw span**, then select **Confirm review** and **Apply**. Enter a reviewer label when you inspect available samples. A gap can retain the decision with evidence unavailable. Human decisions and attestation status appear separately from numerical fit inclusion and the observed lag.

## Hammer-group review

Open **Synthetic example**, select the **Hammer groups** tab, and use **Import group CSV…** with [the example file](../fixtures/hammer-groups.csv). Preview the windows, then select **Apply** to add them to the review or **Cancel preview** to discard the import. Select a group to zoom the waveforms to its UTC window. Use **Next unresolved** to step through the group queue. Choose **Blows** in the navigation filter to visit unresolved calculations and receivers that need current pick review.

The evidence panel compares each declared receiver channel with the embedded channel and reports recorded coverage in the group window. The example includes a recording gap and a channel conflict. A timing or geometry note is a claim to review. **Mark reviewed** records that you inspected the window. **Add blow to group…** starts an individual observation linked to that group. Confirm the association and the arrivals separately.

The CSV uses UTF-8 and requires `group_id` and `start_utc`. Give each row either `end_utc` or `duration_s`, in seconds. Windows include their start and exclude their end. Enter ISO 8601 times with `Z` or `+00:00`. Optional columns are `time_basis`, `source_position_id`, `original_channel`, `calibration_channel`, `timing_note`, and `geometry_note`. Use the receiver columns for the exact expected channel IDs. The review stores each row's source filename, SHA-256 and row number.

Check that applying the import creates zero picks and leaves the existing fit unchanged. Undo, redo, and reopen the session. Verify that the group windows and evidence survive and that a CSV with an invalid or duplicate row produces a line-level error without changing the review. Keep the source log and write the reviewed CSV to a separate file. Version 1 sessions reopen with their existing history and an empty group queue. Version 2 sessions retain their groups and links. Saving a paired review writes version 3. To reopen the upgraded file, use a SeisSlate build that supports version 3.

## Single-channel RMS review

1. Select **File → Open single channel…** and open [the integer reference recording](../fixtures/rms-comparison/integer.mseed). Choose a recorded channel if the file contains more than one.
2. In **Event / Background**, enter event indices `[0, 2)` and background indices `[2, 4)` in the first segment. The stop index is exclusive. You can also double-click the navigation track, drag an interval, and use **Use waveform interval for E** or **Use waveform interval for B**. Read the preview to confirm any snap to the sample grid.
3. Set **Accepted**, enter a reviewer and reason, then select **Preview windows**. Check the exact sample indices and UTC endpoints, two samples per window, RMS values of `2` and `1`, and a contrast of `2`. The fixture has unknown raw units. **Apply** records the decision and calculation. **Cancel preview** leaves the applied result in place.
4. Choose unequal lengths and check that the contrast is unavailable. Cancel. Mark an interval that intersects an applied window, enter a reason, and apply a mask. Check that the previous acceptance no longer applies and the affected result is unavailable. Undo, redo, then save and reopen the review. Use **File → Export RMS review CSV…** to inspect the applied review and replay.

The result is a dimensionless ratio of separately demeaned RMS values over two equal-length, complete windows on one channel. The selected event window can contain background noise. Acquisition epoch and raw units appear independently from the analyst’s review decision and numerical replay.

## Independent RMS reference

Install the desktop extra. Create the ignored `outputs` directory if needed and choose a new output directory for each run.

```sh
python -B tools/compare_measurements.py fixtures/rms-comparison/manifest.json --output-dir outputs/rms-reference
```

The reference reads MiniSEED with Pyrocko and checks the source and sample fingerprints, channel and grid. This reference profile accepts integral sample rates in Hz and retains exact rational sample periods, including at 3 Hz. It computes demeaned population RMS and the event/background ratio independently of the app. Read `sample-manifest.json` for the checked sample support and `reference.csv` for numerical and unavailable outcomes. A manifest case without an expected result produces an unassessed reference. A numerical tolerance in the report checks the implementation. Scientific uncertainty remains separate.

To compare a saved Snuffler marker file, use the named, channel-bound E and B intervals from the included example. Add `--confirm-grid-intent` when both marker intervals were intended to follow exact sample boundaries. Successful matches require unambiguous precision, consistent durations, supported marker fields and exact sample support.

```sh
python -B tools/compare_measurements.py fixtures/rms-comparison/manifest.json --case rational-three-hz --markers fixtures/rms-comparison/snuffler.markers --confirm-grid-intent --output-dir outputs/rms-marker-reference
```

## Relocate a review

Arrange the saved review and its recordings under a common folder. Keep their relative layout, and save a new copy of the session after relinking any recordings that moved. If a retained log is appropriate to share, place it in the same folder tree and identify it explicitly with `--log`. The log has to match a fingerprint already recorded in the review.

```sh
python -m seisslate.handoff create share/handoff.json share/reviews/review.json
python -m seisslate.handoff verify share/handoff.json
```

The manifest lists the session, source recordings, and any supplied logs with relative paths and SHA-256 fingerprints. Move or copy the selected folder to a new location. Run `verify` in a fresh process against the moved manifest, then open the saved review and compare its applied CSV with the original. Missing or changed files produce an error. A verified log fingerprint establishes that the retained bytes match. Its scientific interpretation and the current RMS epoch and units qualification remain separate checks.

## Measure pilot performance

From a Git checkout that contains commit `0f76033`, use the same installed desktop environment as the pilot check. Create the ignored `outputs` directory if needed, then choose a new run directory.

```sh
python -B tools/benchmark_review.py --output-dir outputs/review-benchmark
```

The benchmark records a fresh-process observation and 20 warm observations per measured operation. The workloads include the hammer fixture, one and 24 hours of synthetic waveform data, two one-million-sample windows, and a 1,000-revision review. The current report records the host and Qt backend, dataset preparation, individual timings, process peak memory, session sizes, failures, and a comparison against the pre-RMS hammer workbench. Report results from a native Windows host, Windows emulation, and macOS offscreen as distinct measurements.

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

## Check a paper result against its inputs

Open the [example worksheet](../fixtures/paper-review.csv) in a spreadsheet or text editor. Use one row per check. Record the paper and figure or table, the input locator and SHA-256, the channel and exact sample or time interval, the method and units, the reference and its provenance, the observed result, and the first unresolved dependency. When the row refers to an applied SeisSlate result, record the session file digest and revision ID together. A revision number can recur in another session. Record a numerical tolerance only for comparable values and write down how it was chosen.

These worked examples exercise different steps in a paper. They can be used for practice before a comparison with an independent analyst.

| Example | Check | Result and scope |
| --- | --- | --- |
| **Hengill**, Figure 3 and Table 4 | Request network `2C`, station `BIT06`, and `2019-01-01T21:41:14Z` through `21:41:34Z` from ETH DataSelect. Select `2C.BIT06..HHZ`. The returned file may contain samples outside the query bounds because MiniSEED uses records. Check the decoded grid before selecting an interval. | The verified HHZ component starts at `21:41:13.395Z`, at 200 Hz, with 4,148 samples. To practice raw recomputation, use `[21:41:19Z, 21:41:24Z)` and `[21:41:25Z, 21:41:30Z)`, or sample indices `[1121, 2121)` and `[2321, 3321)` on this file. The worked example gives a demeaned RMS contrast of approximately `0.9601152792032215` from analyst-selected raw windows. The paper illustrates a separate 2–50 Hz filtering comparison. [Article](https://doi.org/10.1038/s41597-022-01339-w) · [Network](https://doi.org/10.12686/sed/networks/2c) · [ETH DataSelect](https://eida.ethz.ch/fdsnws/dataselect/1/) |
| **Global quieting**, Figure 2 and Figure S1 companion notebook | Use `PaperZero_RMS/BE.UCC..HHZ.csv`, the `4.0-14.0` column, and the Brussels time zone in `Supplementary Files/Table S1 - Lockdown stations.csv`. Check the daily median on March 29, 2020, before the notebook's normalization and clipping. | The local-day bin contains 92 observations and has a median of `6.109579592165e-10` in the CSV's input units. Grouping by UTC day instead gives 96 observations and a median of `6.033481540705e-10`. This checks one step using supplied derived data. [Paper](https://doi.org/10.1126/science.abd2438) · [Companion files](https://github.com/ThomasLecocq/2020_Science_GlobalQuieting) |

For a new review, keep the author’s reference, the result of a fresh raw calculation, and the scientific interpretation in separate fields. If an input is unavailable, an instrument response is unverified, or a paper omits a setting needed for comparison, record that condition in the worksheet. A fresh analyst can then resume at the missing step. Use separate held-out cases to evaluate whether linking the worksheet to the waveform helps an analyst find the issue faster.
