# NeuroDataBench

NeuroDataBench is a benchmarking framework for evaluating different data storage, access implementations, object-store backends, and caching strategies for streaming neuroscience data from the cloud. It allows tool maintainers and developers to fairly compare performance across various setups.

## Getting started

### Run an example implementation

From the repository root, run a packaged benchmark with `uv`:

```console
uv run implementations/pynwb_zarr_template.py
```

The `.py` file contains code to fetch data from the files specified in a benchmark (in this case `src/neurodatabench/benchmarks/dynamic_routing_nwb_zarr_v0.json`) and submit answers to the benchmark runner.

Upon completion, profiling results are written to `results/<benchmark_id>/<implementation_id>`
in the current directory. Pass `--out <path>` to use a different output directory.

### Create your own implementation or improve an existing one

- the optional `setup` callable is for creating a common resource that will be used across the data-fetching and answer-submission phases. For example, instantiating a pynwb object for NWB source.
- `submit_answers` is the callable where the actual data fetching and answer submission logic resides:
 iterate over the questions in `context.benchmark.questions`, fetch the data required to answer each question as quickly as possible, in order, and submit the answer for verification with `context.submit_answer(question.id, answer)`.
- explore whether the library being used has alternative data access patterns that are more efficient. For example, the `DynamicTable` class in pynwb provides a `.to_dataframe()` method, with an optional input argument `exclude: set[str]`, which can speed up data fetching and reduce  memory usage if only a subset of columns is needed.

### Create a benchmark
- Create a JSON file defining the benchmark, including the data sources, questions, and expected answers.
- Questions should require fetching data across one or more data sources, representing typical tasks a data scientists would perform when analyzing the dataset.
- They should challenge the data access performance, not the performance of the computation required to produce the answer: minimize the computational overhead once data is in-memory to reduce the opportunity to optimize this component.
- They should describe exactly which table columns to fetch or the slices of arrays to read, in order to get to the required answer.
- The answers must come from the raw data in the data sources, without the possibility of taking shortcuts. For example, if tables are stored in parquet format, row-group metadata for statistics such as min or max could be used to bypass reading the actual data for some questions. Benchmark questions will need to be designed carefully, reviewed and iterated upon!


## Run an implementation matrix

Preview the repository's default matrix without running the actual benchmarks:

```console
uv run --script scripts/run_benchmark_matrix.py --dry-run
```

Custom matrices can reuse the package runner directly:

```python
from pathlib import Path

from neurodatabench.matrix import MatrixRun, run_matrix

runs = [
    MatrixRun(
        implementation="implementations/my_reader.py",
        benchmark="dynamic_routing_nwb_hdf5_v0",
        implementation_id="my-reader.1",
        object_store_backend="s3fs",
        dependencies=("my-reader==1.0",),
    )
]

raise SystemExit(run_matrix(runs, repo_root=Path.cwd()))
```

Note that different versions of python dependencies can be specified in the matrix for comparison.
The `implementation_id` is the run's name used for filtering, logs, and metadata. It may contain
dots; punctuation is sanitized when it is used as a result-directory name.
