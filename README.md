<img src="https://github.com/google/temporian/raw/main/docs/src/assets/banner.png" width="100%" alt="Temporian logo">

[![pypi](https://img.shields.io/pypi/v/temporian?color=blue)](https://pypi.org/project/temporian/)
[![docs](https://readthedocs.org/projects/temporian/badge/?version=stable)](https://temporian.readthedocs.io/en/stable/?badge=stable)
![tests](https://github.com/google/temporian/actions/workflows/test.yaml/badge.svg)
![formatting](https://github.com/google/temporian/actions/workflows/formatting.yaml/badge.svg)
![publish](https://github.com/google/temporian/actions/workflows/publish.yaml/badge.svg)

Temporian is an open-source Python library for preprocessing ⚡ and feature engineering 🛠 temporal data 📈 for machine learning applications 🤖. It is a library tailor-made to address the unique characteristics and complexities of time-related data, such as time-series and transactional data.

> Temporal data is any form of data that represents a state in time. In
> Temporian, temporal datasets contain [events](https://temporian.readthedocs.io/en/stable/user_guide/#events-and-eventsets), which consists of
> values for one or more attributes at a given timestamp. Common
> examples of temporal data are transaction logs, sensor signals, and
> weather patterns. For more, see
> [What is Temporal data](https://temporian.readthedocs.io/en/stable/user_guide/#what-is-temporal-data).

## Key features

- **Unified data processing** 📈: Temporian operates natively on many forms
  of temporal data, including multivariate time-series, multi-index
  time-series, and non-uniformly sampled data.

- **Iterative and interactive development** 📊: Users can easily analyze
  temporal data and visualize results in real-time with iterative tools like
  notebooks. When prototyping, users can iteratively preprocess, analyze, and
  visualize temporal data in real-time with notebooks. In production, users
  can easily reuse, apply, and scale these implementations to larger datasets.

- **Avoids future leakage** 😰: Future leakage occurs during model training
  when a model is exposed to data from future events, which leaks information
  that would otherwise not be available to the model and can result in
  overfitting. Temporian operators do not create leakage by default. Users
  can also use Temporian to programmatically detect whether specific signals
  were exposed to future leakages.

- **Flexible runtime** ☁️: Temporian programs can run seamlessly in-process in
  Python, on large datasets using [Apache Beam](https://beam.apache.org/).

- **Highly optimized** 🔥: Temporian's core is implemented and optimized in
  C++, so large amounts of data can be handled in-process. In some cases,
  Temporian is 1000x faster than other libraries.

> **Note**
> Temporian's development is in alpha.

## QuickStart

### Installation

Temporian is available on [PyPI](https://pypi.org/project/temporian/). Install it with pip:

```shell
pip install temporian
```

### Minimal example

The following example uses a dataset, `sales.csv`, which contains transactional data. Here is a preview of the data:

```shell
$ head sales.csv
timestamp,store,price,count
2022-01-01,CA,27.42,61.9
2022-01-01,TX,98.55,18.02
2022-01-02,CA,32.74,14.93
2022-01-15,TX,48.69,83.99
...
```

The following code calculates the weekly sales for each store, visualizes the output with a plot, and exports the data to a CSV file.

```python
import temporian as tp

input_data = tp.from_csv("sales.csv")

per_store = input_data.set_index("store")
weekly_sum = per_store["price"].moving_sum(window_length=tp.duration.days(7))

# Plot the result
weekly_sum.plot()

# Save the results
tp.to_csv(weekly_sum, "store_sales_moving_sum.csv")
```

![](https://github.com/google/temporian/raw/main/docs/src/assets/frontpage_plot.png)

Check the [Getting Started tutorial](https://temporian.readthedocs.io/en/stable/tutorials/getting_started/) to try it out!

## Next steps

New users should refer to the [Getting Started](https://temporian.readthedocs.io/en/stable/getting_started/) guide, which provides a
quick overview of the key concepts and operations of Temporian.

After that, visit the [User Guide](https://temporian.readthedocs.io/en/stable/user_guide/) for a deep dive into
the major concepts, operators, conventions, and practices of Temporian. For a
hands-on learning experience, work through the [Tutorials](https://temporian.readthedocs.io/en/stable/tutorials/) or refer to the [API
reference](https://temporian.readthedocs.io/en/stable/reference/).

## Documentation

The documentation 📚 is available at [temporian.readthedocs.io](https://temporian.readthedocs.io/en/stable/). The [Getting Started guide](https://temporian.readthedocs.io/en/stable/getting_started/) is the best way to start.

## Contributing

Contributions to Temporian are welcome! Check out the [contributing guide](CONTRIBUTING.md) to get started.

## Credits

Temporian is developed in collaboration between Google and [Tryolabs](https://tryolabs.com/).
