<img src="https://github.com/google/temporian/blob/main/docs/src/assets/banner.png?raw=true" width="100%" alt="Temporian logo">

![tests](https://github.com/google/temporian/actions/workflows/test.yaml/badge.svg) ![formatting](https://github.com/google/temporian/actions/workflows/formatting.yaml/badge.svg) [![docs](https://readthedocs.org/projects/temporian/badge/?version=latest)](https://temporian.readthedocs.io/en/latest/?badge=latest)

**Temporian** is a Python package for **feature engineering of temporal data**, focusing on **preventing common modeling errors** and providing a **simple and powerful API**, a first-class **iterative development** experience, and **efficient and well-tested implementations** of common and not-so-common temporal data preprocessing functions.

## Installation

Temporian is available on PyPI. To install it, run:

```shell
pip install temporian
```

## Getting Started

This is how a minimal end-to-end example looks like:

```python
import temporian as tp

# Load data and create input node.
evset = tp.from_csv("temporal_data.csv", timestamp_column="date")
source = evset.node()

# Apply operators to create a processing graph.
sma = tp.simple_moving_average(source, window_length=tp.duration.days(7))

# Run the graph.
result_evset = sma.evaluate({source: evset})

# Show output.
print(result_evset)
result_evset.plot()
```

This is an example `temporal_data.csv` to use with the code above:

```
date,feature_1,feature_2
2023-01-01,10.0,3.0
2023-01-02,20.0,4.0
2023-02-01,30.0,5.0
```

Check the [Getting Started tutorial](https://temporian.readthedocs.io/en/stable/tutorials/getting_started/) to try it out!

## Key features

These are what set Temporian apart.

- **Simple and powerful API**: Temporian exports high level operations making processing complex programs short and ready to read.
- **Flexible data model**: Temporian models temporal data as a sequence of events, supporting non-uniform sampling timestamps seamlessly.
- **Prevents modeling errors**: Temporian programs are guaranteed not to have future leakage unless explicitly specified, ensuring that models are not trained on future data.
- **Iterative development**: Temporian can be used to develop preprocessing pipelines in Colab or local notebooks, allowing users to visualize results each step of the way to identify and correct errors early on.
- **Efficient and well-tested implementations**: Temporian contains efficient and well-tested implementations of a variety of temporal data processing functions. For instance, our implementation of window operators is **x2000** faster than the same function implemented with NumPy.
- **Wide range of preprocessing functions**: Temporian contains a wide range of preprocessing functions, including moving window operations, lagging, calendar features, arithmetic operations, index manipulation and propagation, resampling, and more. For a full list of the available operators, see the [operators documentation](https://temporian.readthedocs.io/en/stable/reference/).

## Documentation

The official documentation is available at [temporian.readthedocs.io](https://temporian.readthedocs.io/en/stable/).

## Contributing

Contributions to Temporian are welcome! Check out the [contributing guide](CONTRIBUTING.md) to get started.

## Credits

This project is a collaboration between Google and [Tryolabs](https://tryolabs.com/).
