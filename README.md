<img src="https://github.com/google/temporian/raw/main/docs/src/assets/banner.png" width="100%" alt="Temporian logo">

[![pypi](https://img.shields.io/pypi/v/temporian?color=blue)](https://pypi.org/project/temporian/)
[![docs](https://readthedocs.org/projects/temporian/badge/?version=stable)](https://temporian.readthedocs.io/en/stable/?badge=stable)
![tests](https://github.com/google/temporian/actions/workflows/test.yaml/badge.svg)
![formatting](https://github.com/google/temporian/actions/workflows/formatting.yaml/badge.svg)
![publish](https://github.com/google/temporian/actions/workflows/publish.yaml/badge.svg)

**Temporian** is a library for **safe**, **simple** and **efficient** preprocessing and feature engineering of temporal data in Python. Temporian supports multivariate time-series, multivariate time-sequences, event logs, and cross-source event streams.

Temporian is to [temporal data](https://temporian.readthedocs.io/en/stable/user_guide/#what-is-temporal-data) what Pandas is to tabular data.

## Key features

- **Supports most types of temporal data** 📈: Handles both uniformly sampled and
  non-uniformly sampled data, both single-variate and multivariate data, both flat
  and multi-index data, and both mono-source and multi-source non-synchronized
  events.

- **Optimized for Temporal data** 🔥: Temporian's core computation is
  implemented in C++ and optimized for temporal data. Temporian can be more than
  1,000x faster than off-the-shelf data processing libraries when operating on
  temporal data.

- **Easy to integrate into an existing ML ecosystem**: Temporian does not perform any ML model training - instead it integrates seamlessly with any ML library, such as PyTorch, Scikit-Learn, Jax, TensorFlow, XGBoost, or Yggdrasil Decision Forests.

- **Prevents unwanted future leakage** 😰: Unless explicitly specified with
  `tp.leak`, feature computation cannot depend on future data, thereby preventing
  unwanted, hard-to-debug, and potentially costly future leakage.

<!--
- **Iterative and interactive development** 📊: Users can easily analyze
  temporal data and visualize results in real-time with iterative tools like
  notebooks. When prototyping, users can iteratively preprocess, analyze, and
  visualize temporal data in real-time with notebooks. In production, users
  can easily reuse, apply, and scale these implementations to larger datasets.

- **Flexible runtime** ☁️: Temporian programs can run seamlessly in-process in
  Python, on large datasets using [Apache Beam](https://beam.apache.org/).
-->

## Quickstart

### Installation

Install Temporian from [PyPI](https://pypi.org/project/temporian/) with `pip`:

```shell
pip install temporian -U
```

Temporian is currently available for Linux and MacOS (ARM and Intel). Windows support is under development.

### Minimal example

Consider sale records that contain contain the `timestamp`, `store`, and `revenue` of individual sales.

```shell
$ cat sales.csv
timestamp,store,revenue
2023-12-04 21:21:05,STORE_31,5071
2023-11-08 17:14:38,STORE_4,1571
2023-11-29 21:44:46,STORE_49,6101
2023-12-20 18:17:14,STORE_18,4499
2023-12-15 10:55:09,STORE_2,6666
...
```

Our goal is to compute the sum of revenue for each store at 11 pm every weekday (excluding weekends).

First, we load the data and list the workdays.

```python
import temporian as tp

# Load sale transactions
sales = tp.from_csv("sales.csv")

# Index sales per store
sales_per_store = sales.add_index("store")

# List work days
days = sales_per_store.tick_calendar(hour=22)
work_days = (days.calendar_day_of_week() <= 5).filter()

work_days.plot(max_num_plots=1)
```

![](https://github.com/google/temporian/raw/main/docs/src/assets/frontpage_workdays.png)

Then, we sum the daily revenue for each workday and each store.

```python
# Aggregate revenue per store and per work day
daily_revenue = sales_per_store["revenue"].moving_sum(tp.duration.days(1), sampling=work_days).rename("daily_revenue")

# Plot the results
daily_revenue.plot(max_num_plots=3)
```

![](https://github.com/google/temporian/raw/main/docs/src/assets/frontpage_aggregated_revenue.png)

Finally, we can export the result as a Pandas DataFrame for further processing or for consumption by other libraries.

```python
tp.to_pandas(daily_revenue)
```

![](https://github.com/google/temporian/raw/main/docs/src/assets/frontpage_pandas.png)

Check the [Getting Started tutorial](https://temporian.readthedocs.io/en/stable/tutorials/getting_started/) to find out more!

## Next steps

New users should refer to the [Getting Started](https://temporian.readthedocs.io/en/stable/getting_started/) guide, which provides a
quick overview of the key concepts and operations of Temporian.

After that, visit the [User Guide](https://temporian.readthedocs.io/en/stable/user_guide/) for a deep dive into
the major concepts, operators, conventions, and practices of Temporian. For a
hands-on learning experience, work through the [Tutorials](https://temporian.readthedocs.io/en/stable/tutorials/) or refer to the [API
reference](https://temporian.readthedocs.io/en/stable/reference/).

If you need help, have a question, want to contribute, or just want to be a part of the Temporian community, we encourage you to join our [Discord](https://discord.gg/nT54yATCTy) server! 🤝🏼

## Documentation

The documentation 📚 is available at [temporian.readthedocs.io](https://temporian.readthedocs.io/en/stable/). The [Getting Started guide](https://temporian.readthedocs.io/en/stable/getting_started/) is the best way to start.

## Contributing

Contributions to Temporian are welcome! Check out the [Contributing guide](https://temporian.readthedocs.io/en/stable/contributing/) to get started.

## Credits

Temporian is developed in collaboration between Google and [Tryolabs](https://tryolabs.com/).
