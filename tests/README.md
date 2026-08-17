# Tests

Plain `unittest`, no extra dependency. From the repository root:

    python3 -m unittest discover -s tests            # everything
    python3 -m unittest discover -s tests -v         # with test names

What each file needs:

| file | needs | runtime |
|---|---|---|
| `test_config.py` | nothing | instant |
| `test_pomdp_core.py` | nothing | seconds |
| `test_determinism.py` | nothing | instant |
| `test_make_figures.py` | nothing | instant |
| `test_run_pipeline.py` | nothing | instant |
| `test_preprocessing.py` | the dataset in `data/TrHu_NHB_light/` | seconds |
| `test_reproducibility.py` | the deposited `results/` | seconds |

Tests skip rather than fail when their inputs are missing, so a fresh clone with
no data still runs the files that need nothing and reports the rest as skipped.
That is the quickest way to tell whether an install is working. The three that
check the runners import nothing from the scientific stack, so they still run
when that stack is itself what is broken.

`test_determinism.py` guards the seeding. A seed taken from the builtin `hash()`
of a string reads as deterministic and changes on every run, because Python
salts that hash per process.

`test_reproducibility.py` checks the numbers quoted in the manuscript against
the deposited fits and tables. If one of those fails after a change, the change
altered a published result.
