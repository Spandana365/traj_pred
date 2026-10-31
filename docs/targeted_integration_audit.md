# Targeted Integration Audit

**Purpose.** Record what the running dashboard actually shows, before any change, so that working ETH integrations are preserved and only mock, incomplete or incorrect sections are replaced.

**Method.** Every route was driven in a headless Chromium browser against the running servers at `http://localhost:3000` and `http://localhost:8000`, at 1440x900. The rendered text, charts, images, links and console were recorded, and every button in `main` was clicked to see whether anything changed. Findings were then checked against the source code and saved artifacts.

**Status of this document.** Written before any code was changed. Section 6 records what was done afterwards.

**Status terms**

- **Integrated:** rendered from a real saved artifact through the API.
- **Partial:** real data, but incomplete or with a defect.
- **Mock:** labelled as mock, or showing illustrative content.
- **Placeholder:** an empty structure with no data behind it.
- **Non-functional:** a control that does nothing.
- **Missing:** the saved artifact exists but is not shown.

---

## 1. Summary

| Route | Initial state | Component rendered | Mock banner |
|---|---|---|---|
| `/` | Redirects to `/overview` | `app/page.tsx` | No |
| `/overview` | ETH integrated; MOT17 missing | `OverviewResultsPage` | No |
| `/analysis` | ETH tables integrated; **both charts blank** | `AnalysisResultsPage` | No |
| `/data` | ETH integrated; MOT17 missing | `DataResultsPage` | No |
| `/experiments` | ETH integrated; MOT17 missing | `ExperimentsResultsPage` | No |
| `/tracking` | Stub saying "evaluation pending" | `TrackingResultsPage` | No |
| `/error-injection` | Mock, no real results | `ErrorInjectionPage` | **Yes** |
| `/error-visualization` | Mock, with a sample timeline | `ErrorVisualizationPage` | **Yes** |
| `/prediction` | Mock, with illustrative curves | `PredictionPage` | **Yes** |
| `/application-impact` | Placeholder, no data exists | `ImpactPage` | **Yes** |
| `/settings` | Connection is real; other controls do nothing | `SettingsPage` | No |

No route produced a console error. The only broken route is `/experiments/TV-2026-001`, which returns 404. No page links to it; only the unused `ExperimentsPage` component did.

### ETH verification before any change

An independent script derived expected values directly from the saved seed reports and baseline report, without using the API. All **322 of 322** values and images were present in the rendered pages:

- `/overview`: both baselines, window count, and the full 12-row error table.
- `/data`: all seven per-scene rows.
- `/analysis` and `/experiments`: all 12 experiment and severity combinations, including every per-scene row.
- `/experiments`: the image for each of the 12 combinations.

---

## 2. Page audit

| Page | Section/component | Current status | Data source | Required action |
|---|---|---|---|---|
| All | Sidebar navigation | Integrated | Static links | Preserve |
| All | API connection badge | Integrated | `/api/v1/health` | Preserve |
| All | Header "Workspace draft / TV-2026-001 / ETH / Hotel" | Mock | Hardcoded `createInitialExperiment` | Replace; it wrongly implies one ETH scene is selected |
| All | Header Save and Reset buttons | Non-functional | Local state only | Remove |
| All | Workflow step counter on `/experiments` and `/settings` | Partial | Shows "00/08" | Correct |
| Overview | ETH metric cards and baselines | Integrated | `dashboard/summary`, `prediction/baseline` | Preserve |
| Overview | ETH error-injection table | Integrated | `error-injections` | Preserve |
| Overview | MOT17 tracking summary | Missing | `gmc_fixed/trackeval` | Integrate |
| Analysis | ADE and FDE by severity charts | **Partial: blank** | Real data, zero-height container | Correct (CSS class) |
| Analysis | Experiment and severity filters | Integrated | `error-injections/{experiment}` | Preserve |
| Analysis | ETH detail, per-scene and per-timestep tables | Integrated | `error-injections/{experiment}` | Preserve |
| Analysis | MOT17 tracking analysis | Missing | `gmc_fixed/error_profiles` | Integrate as a separate section |
| Data | ETH metric cards, scenes and per-scene table | Integrated | `dashboard/summary`, `prediction/baseline` | Preserve; retitle page because it will cover two datasets |
| Data | ETH dataset audit | Missing | `runs/audits/eth/dataset-audit.json` | Integrate |
| Data | MOT17 sequence inventory | Missing | `config/dataset_inventory.json` | Integrate |
| Experiments | ETH summary table, filters, detail and image | Integrated | `error-injections` | Preserve |
| Experiments | Per-seed ETH results | Partial | API supports `seed`; no control exposes it | Add a seed filter |
| Experiments | MOT17 tracking experiments | Missing | `gmc_fixed/` | Integrate as a separate section |
| Tracking | "Tracker evaluation pending", ByteTrack and BoT-SORT cards | Placeholder | Text only | Replace with MOT17 results; ByteTrack was not implemented |
| Error injection | Five error-family cards with severity sliders | Mock | Hardcoded `errorFamilies` | Replace with the four real ETH experiments |
| Error injection | "Ego-Motion Error" family | Mock | No experiment exists | Remove |
| Error injection | Experiment summary ("Placeholder", "0.20", "—") | Mock | Hardcoded | Replace |
| Error injection | Preview error, Run injection, Reset | Non-functional | None | Remove; the API is read-only |
| Error visualization | Original and error-controlled video viewers | Placeholder | None; no videos exist | Replace with saved figures |
| Error visualization | Play, Frame step, Sync, Fullscreen | Non-functional | None | Remove |
| Error visualization | Error event timeline | **Mock** | `mockEvents`, under a "Backend-generated events" label | Remove; no such artifact exists |
| Error visualization | Error summary ("—") | Placeholder | None | Replace with real figures |
| Error visualization | MOT17 tracking examples | Missing | `gmc_fixed/visualizations` | Integrate |
| Prediction | Clean and corrupted trajectory graphics | **Mock** | Hardcoded SVG curves | Remove |
| Prediction | Model, history and horizon controls | Non-functional | No effect | Remove; these do not run anything |
| Prediction | Constant Acceleration and GRU options | Mock | Neither model exists | Remove |
| Prediction | ADE, FDE, RMSE and Coverage cards | Placeholder | "—" | Replace with real ETH baselines |
| Prediction | ETH per-timestep and per-scene baseline | Missing | `prediction/baseline` | Integrate; API already returns it |
| Application impact | Mock banner | Mock | No research values are shown | Replace with an unavailable notice |
| Application impact | Metric table, all "—" | Placeholder | No artifact exists | Keep unavailable; no results to connect |
| Application impact | Interpretation boundary | Integrated (text) | Static | Preserve |
| Settings | API endpoint and connection status | Integrated | `/api/v1/health` | Preserve |
| Settings | Default detector, tracker, Developer mode | Non-functional | Local only; shows "YOLO · version pending" | Replace with read-only MOT17 configuration |
| Settings | Theme and overlay selects | Non-functional | Single options | Remove |

---

## 3. Mock and hardcoded data

| Item | File | Rendered? | Assessment | Action |
|---|---|---|---|---|
| `mockEvents` | `types/index.ts` | Yes, `/error-visualization` | Fabricated events | Remove |
| `TrajectoryGraphic` curves | `trackvision-app.tsx` | Yes, `/prediction` | Illustrative | Remove |
| `errorFamilies` | `types/index.ts` | Yes, `/error-injection` | Includes an unimplemented family | Remove |
| `demoIndicator`, `DemoNotice` | Both | Yes, four routes | Correct until those pages are connected | Remove after connection |
| `createInitialExperiment` | `types/index.ts` | Yes, header | Invented experiment ID and scene | Stop displaying it |
| `mockDatasets` | `types/index.ts` | No, unused `DataPage` | Fabricated metadata, including MOT17-04 as a moving camera | Remove |
| `mockExperiments` | `types/index.ts` | No | **Fabricated "completed" BoT-SORT + LSTM on MOT17-04** | Remove |
| `mockChartSeries` | `types/index.ts` | No, unused `AnalysisPage` | Null placeholder values | Remove |
| `createMockTrajectory` | `types/index.ts` | No | Fabricated trajectories | Remove |
| Inline experiment rows | `ExperimentsPage` | No | **Fabricated MOT17 + LSTM + dropout history** | Remove |
| `OverviewPage`, `DataPage`, `TrackingPage`, `AnalysisPage`, `ExperimentsPage`, `ExperimentDetailPage`, `DashboardPage` | `trackvision-app.tsx` | No | Unused predecessors with mock content | Remove |
| `workflowStages`, status types, `formatMetric` | `types/index.ts` | Partly | Harmless definitions | Leave unless unused imports need tidying |
| `EXPERIMENTS`, `SEVERITIES` | `research-results.tsx` | Yes | Legitimate labels for API IDs | Preserve |
| Scientific note text | `research-results.tsx` | Yes | Legitimate | Preserve |

---

## 4. Stylesheet defects

`web/app/globals.css` is a compiled Tailwind v4 file with no `@import "tailwindcss"`, so a class used in source does nothing unless it is already compiled in. Fourteen real classes used by working pages were undefined:

| Class | Effect | Visible consequence |
|---|---|---|
| `h-[280px]` | Chart height | **Both `/analysis` charts render at zero height** |
| `min-w-[520px]`, `min-w-[900px]` | Table width | Tables compress instead of scrolling |
| `max-h-[560px]`, `max-w-full`, `object-contain` | Image limits | `/experiments` images are unconstrained |
| `max-h-80`, `overflow-auto` | Scrolling | Per-timestep tables do not scroll in place |
| `text-amber-200`, `text-amber-200/80`, `border-amber-400/40` | Note styling | Unit and method notes lose their intended colour |
| `pl-3`, `pr-3` | Spacing | Tighter table columns |
| `break-words`, `capitalize` | Text | Long values and severity labels unformatted |

**Action:** add these rules, plus any new ones, to the existing utilities layer. Do not regenerate the stylesheet, because that could change styling across the application.

---

## 5. Verified artifact status

### ETH

| Result | Source artifact | API | Frontend | Classification |
|---|---|---|---|---|
| Clean baselines | `prediction/eth/lstm-seed42-h64-l2-scheduler/validation_results.json` | `prediction/baseline`, `dashboard/summary` | `/overview`, `/data` | Fully integrated and verified |
| LSTM configuration | `.../experiment.json` | `dashboard/summary` | Not displayed | Partially integrated |
| Per-timestep baseline | `validation_results.json` | `prediction/baseline` | Not displayed | Partially integrated |
| Localization noise | `error_injection/localization_noise/` | `error-injections` | `/overview`, `/analysis`, `/experiments` | Fully integrated and verified |
| Detection dropout | `error_injection/detection_dropout/` | Same | Same | Fully integrated and verified |
| Track fragmentation | `error_injection/track_fragmentation/` | Same | Same | Fully integrated and verified |
| Identity switches | `error_injection/identity_switch/` | Same | Same | Fully integrated and verified |
| Severity charts | Same reports | Same | `/analysis` | Integrated but requiring a small correction |
| Per-seed results | `per_seed/seed-*.json` | `?seed=` | Aggregates only | Partially integrated |
| Example figures | `*-example-*.png` | `visualizations/{severity}` | `/experiments` | Fully integrated and verified |
| Summary figures and LSTM comparison plots | `*.png`, `lstm_comparison_plots/` | None | None | Not integrated |
| Dataset audit | `audits/eth/dataset-audit.json` | None | None | Not integrated |
| Sealed test result | `.../test-results.json` | None | None | **Deliberately not integrated** |

`test-results.json` holds a limited held-out-scene evaluation. It stays out of the dashboard so validation and sealed test metrics cannot be mixed.

### MOT17

No MOT17 result was available from the API or shown on any route.

| Item | Authoritative artifact |
|---|---|
| Sequence inventory | `runs/mot17_tracking_analysis/config/dataset_inventory.json` |
| Detector and tracker configuration | `gmc_fixed/config/run_config.json`, `config/weights_fingerprint.json` |
| Camera motion | `gmc_fixed/gmc/verification.json` |
| GMC diagnostics | `gmc_fixed/gmc/diagnostics.json` |
| Control run | `gmc_fixed/control_no_gmc/control_result.json` |
| Primary TrackEval metrics | `gmc_fixed/trackeval/BoT-SORT_GMC_enabled/pedestrian_detailed.csv` |
| Ablation TrackEval metrics | `trackeval/botsort/pedestrian_detailed.csv` |
| Error profiles | `gmc_fixed/error_profiles/` and `error_profiles/` |
| Figures | `gmc_fixed/visualizations/` |

`camera_motion.json` says GMC "stays on", which the audit showed was false for the archived run, so the dashboard uses `gmc_fixed/gmc/verification.json` instead.

### Limits on what can be shown

- **No MOT17 prediction results exist**, so none are shown.
- **No application-impact artifact exists**, so that page stays unavailable.
- **No videos exist**, so viewers are replaced by saved figures.
- **No ETH severity is MOT17-calibrated.** No calibration artifact exists.
- **ByteTrack was not implemented.**

---

## 6. Changes made after this audit

### Preserved unchanged

ETH metric cards, baselines, the error-injection table, severity charts, filters, per-scene and per-timestep tables, experiment details and example images all keep their code and design. The only edits around them are section headings, the seed filter and the stylesheet fixes in section 4. All existing API endpoints are unchanged, and the 41 original API tests pass unmodified.

### Corrected

| Page | Change |
|---|---|
| `/analysis` | Both ETH severity charts now render, through the stylesheet fix |
| All | Fourteen missing utility classes added to `globals.css` |
| All | Fabricated "TV-2026-001 / ETH / Hotel" header replaced; Save and Reset removed |
| `/experiments`, `/settings` | Workflow counter shows "—/08" instead of "00/08"; sidebar highlights the current link |
| `/data`, `/analysis`, `/experiments` | Titles changed because each page now covers two labelled studies |

### Replaced

| Page | Before | After |
|---|---|---|
| `/tracking` | "Evaluation pending" stub | Full MOT17 results |
| `/error-injection` | Five mock families and dead buttons | Four real ETH experiments with severity results |
| `/error-visualization` | Empty viewers and fake timeline | Saved ETH figures and real MOT17 tracking figures |
| `/prediction` | Illustrative curves and dead controls | ETH validation baselines, configuration, timestep chart and per-scene comparison |
| `/settings` | Dead selects | Real connection status and saved MOT17 configuration |
| `/application-impact` | Mock banner | Explicit "not measured" notice; no values were added |

### Added

- **ETH:** per-seed filter, dataset audit panel, and 12 saved summary figures.
- **MOT17 overview:** combined summary, labelled separately from ETH.
- **MOT17 data:** seven-sequence inventory.
- **MOT17 tracking:** combined and per-sequence metrics, configuration, GMC diagnostics, ablation comparison and the MOT17-11 regression.
- **MOT17 analysis:** localization, track duration, gaps, coverage and availability, with an experiment filter.
- **MOT17 experiments:** primary and comparison experiments.
- **MOT17 visualization:** seven sequence frames and three figures for each experiment.

### Removed

All unused mock pages and data listed in section 3: `mockDatasets`, `mockExperiments`, `mockChartSeries`, `mockEvents`, `createMockTrajectory`, `demoIndicator`, `errorFamilies`, `DemoNotice` and the seven predecessor pages. The default state in `createInitialExperiment` no longer names an invented experiment.

### API

**Reused unchanged:** `health`, `dashboard/summary`, `prediction/baseline`, `error-injections`, `error-injections/{experiment}` (including `?seed=`) and `error-injections/{experiment}/visualizations/{severity}`.

**Added for ETH:** `GET /api/v1/datasets/eth/dataset-audit`, `GET /api/v1/datasets/eth/error-injections/figures`, and `GET /api/v1/datasets/eth/error-injections/figures/{figure_id}`.

**Added for MOT17, under `/api/v1/datasets/mot17/tracking/`:** `summary`, `sequences`, `configuration`, `gmc`, `error-profile`, `ablation`, and `visualizations/{figure}`.

`src/trackvision/api/mot17_repository.py` is separate from the ETH repository. It reads fixed paths only, and it refuses to serve the primary run unless its saved status and diagnostics show completion and its two metric files agree. Missing artifacts return 404 and no values.

### Remaining incomplete

| Item | Status |
|---|---|
| Application impact | No artifact exists; the page says so |
| MOT17 prediction | Not implemented; `/prediction` says so |
| Sealed ETH test results | Deliberately not displayed |
| MOT17-calibrated severities | No calibration artifact exists; labelled as exploratory |
| ByteTrack | Not implemented, so not displayed |
| Video viewers | No videos were saved, so saved figures are used |
| Sidebar "Workflow" link | Goes to `/overview`, as "Dashboard" does; left unchanged |
| Unused `Tracker` and `isMock` type names | Harmless definitions; nothing renders them |
| MOT17-11 mechanism | Likely explanation only; false tracks were not checked by hand |

Automated checks ran against a headless browser, so the final look in your own browser was not checked.

