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

## Review signal boundaries at Ellsworth Glacier

The landslide exercise reviews the observable start and end of a signal on one channel in one frequency band. The 2022 Ellsworth Glacier event comes from the USGS authors’ software example. Its reference table contains manual annotations for 1–5 Hz and 20–100 s. The event spans shown in the paper and example help locate data. Each annotation belongs to its station and band. A missing row does not establish observation, coverage, or absence of a signal. [USGS software release](https://doi.org/10.5066/P13NSNIZ) · [Companion data catalog](https://doi.org/10.5066/P13FCYQO)

### Prepare the waveform sources

The [Ellsworth case manifest](../fixtures/ellsworth-landslide.json) records three waveform checksums, the source queries, and the inspected channel epochs. Download the three MiniSEED files from their DataSelect URLs in the manifest and save them under `data/ellsworth/` with the listed names. The waveform files and generated review packets live in local ignored folders. The original downloads are single 50 Hz traces with at least 613 seconds beyond either end of the review interval. The exercise withholds `AK.Q23K..BHZ` as an explicit unavailable-data control. Its archived waveform exists, so the omitted trace does not say anything about whether the event was observed there. [Alaska Geophysical Network](https://doi.org/10.7914/SN/AK) · [EarthScope DataSelect](https://service.earthscope.org/fdsnws/dataselect/1/)

A coordinator can download the [pinned author CSV](https://code.usgs.gov/api/v4/projects/18158/repository/files/EXAMPLE.csv/raw?ref=400ff9d6c9ac94363b3be1412c6d2fd6f210d8ed) and retain it separately until review is complete. The author CSV checksum in the case manifest matches the software release at tag `1.0.0`. This case uses the software example. The separate global ScienceBase attachment could not be retrieved for this run.

### Run both review paths

From the checkout, use the installed desktop environment and an existing `outputs` directory. Preparation writes a new packet with checksummed copies of the original files, a blank review table, and MiniSEED traces for stock Snuffler. Both paths use the same 450-second review interval.

```sh
python -m tools.landslide_pilot prepare fixtures/ellsworth-landslide.json outputs/ellsworth-pack
python -m tools.landslide_pilot worksheet outputs/ellsworth-pack outputs/ellsworth-worksheet.json
python -m tools.landslide_pilot snuffle outputs/ellsworth-pack outputs/ellsworth-review.markers
```

In the worksheet, select a channel and band, choose Onset or End, and click in the corresponding waveform to preview a pick. Use **−1 sample** or **+1 sample** to refine the preview, then enter a reason and choose **Apply**. For an unresolved boundary, choose its review status and give a reason. Use **Undo** to recover the earlier applied decision and save the result.

For the comparator, copy `outputs/ellsworth-pack/review.csv` to `outputs/ellsworth-table.csv`. In Snuffler, double-click the named band trace to enter picking mode and place a plain point marker. Select the marker and press **1** for onset or **2** for end. Use one marker for each `identifiable` boundary. Leave markers off unresolved boundaries and record their choices and reasons in the table. Keep the prepared `unavailable` and `insufficient_support` states. Keep the supplied waveform processing settings. Close Snuffler to export its markers, then score the completed table with the reference calculation. [Snuffler picking controls](https://pyrocko.org/docs/current/apps/snuffler/manual.html#keystrokes)

```sh
python -m tools.landslide_pilot score outputs/ellsworth-pack outputs/ellsworth-table.csv outputs/ellsworth-review.markers outputs/ellsworth-baseline.json
python -m tools.landslide_pilot reopen outputs/ellsworth-pack outputs/ellsworth-baseline.json
python -m tools.landslide_pilot reopen outputs/ellsworth-pack outputs/ellsworth-worksheet.json
```

Supply the held-back CSV only after saving each reviewer’s first pass. The reveal records the original author values separately, including their timestamp precision.

```sh
python -m tools.landslide_pilot reveal outputs/ellsworth-pack outputs/ellsworth-worksheet.json data/ellsworth/EXAMPLE.csv outputs/ellsworth-author-reference.json
```

The pilot filters original recording units with a four-corner ObsPy Butterworth bandpass, forward and backward with zero initial states. It applies no response correction, taper or extra padding. The support rule withholds the first and last 2 seconds of each uninterrupted filter input for the high-frequency band and 200 seconds for the long-period band. Those guards define where a pick is eligible. The filtered samples still depend on the entire uninterrupted input. The packet records the filter, source and output fingerprints and the versions used for exact replay.

### Compare with Snuffler and ObsPy

Give both paths the same raw sources, analysis span, processed waveform samples and status choices. Start each independent reviewer with a fresh worksheet and keep their first saved files for comparison. Keep the authors’ annotations outside the first-pass packet. Use different matched cases and balance which interface each analyst uses first. Give both paths equivalent instructions, practice, and a prepared handoff workflow.

1. Identify a clear signal and mark each visible boundary. Review another row where one boundary is uncertain. Preserve the visible boundary and give a reason for the other decision. Exercise a gap or unsupported edge as a separate control.
2. Change an applied boundary, verify the change in the observed duration and save the reviewed state. Have a second analyst reopen the result and recover the channel and band, original sample indices, review reasons and measurement eligibility. Reveal the authors’ annotations afterward. Record differences in picks separately from failures to recover the measurements.
3. Time preparation, review, correction and independent reopening. Count all assigned cases, critical errors, false-ready outcomes, missed eligible results and timeouts. One case can demonstrate feasibility. Use fresh, counterbalanced cases before estimating a repeatable time benefit. The exploratory target is at least 20% lower median within-reviewer total time with no increase in consequential errors. A go decision requires zero critical channel, band, sample, duration, review state or provenance errors.

Use the worksheet duration only when two ordered boundaries have support in the same uninterrupted processed segment. A conclusion of no identifiable signal requires complete eligible support through the requested review interval. An unresolved boundary leaves duration unavailable. The exercise measures the visible signal at that channel and band. The USGS study uses high-frequency duration and long-period maximum amplitude as different features. Technical replay does not reproduce the authors’ processing or measure source duration, volume, or detection accuracy. [Author picking and processing example](https://code.usgs.gov/ghsc/lhp/reference/landslide-volume-estimation/-/tree/1.0.0)

### Ellsworth technical result

In a scripted comparison on macOS with offscreen Qt, stock Snuffler loaded nine traces from the three channels. Pyrocko read back the same samples supplied to the worksheet for all six filtered traces. The two review paths agreed on all eight channel-and-band rows, including two unavailable rows and two one-sided decisions. Moving a scripted onset forward by one sample at 50 Hz shortened its interval by 0.02 s. Both reviews reopened from the relocated packet and the original waveform fingerprints still matched. The results establish technical agreement for the scripted decisions. The independent analyst comparison and native Windows exercise remain pending.
