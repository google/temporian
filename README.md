<img src="https://github.com/google/temporian/blob/main/docs/src/assets/banner.png?raw=true" width="100%" alt="Temporian logo">

![tests](https://github.com/google/temporian/actions/workflows/test.yaml/badge.svg) ![formatting](https://github.com/google/temporian/actions/workflows/formatting.yaml/badge.svg)

**Temporian** is a Python package for **feature engineering of temporal data**, focusing on providing a **simple and powerful API**, a first-class **iterative development** experience, **efficient and well-tested implementations** of common and not-so-common temporal data preprocessing functions, and **preventing common modeling errors**.

## ¿Why Temporian?

Temporian helps you **focus on high-level modeling**.

Temporal data processing is commonly done with generic data processing tools. However, this approach is often tedious, error-prone, and requires engineers to learn and re-implement existing methods. Additionally, the complexity of these tools can lead engineers to create less effective pipelines in order to reduce complexity. This can increase the cost of developing and maintaining performant ML pipelines.

To see the benefit of Temporian over general data processing libraries, compare the original **Feature engineering** section of our [Khipu 2023 Forecasting Tutorial](https://github.com/tryolabs/khipu-2023), which uses pandas to preprocess the M5 sales dataset, to the [updated version using Temporian](docs/src/tutorials/m5_competition.py).

## Installation

Temporian is available on PyPI. To install it, run:

```shell
pip install temporian
```

## Minimal end-to-end example

```python
import temporian as tp

# Load data.
evset = tp.read_event_set("path/to/temporal_data.csv", timestamp_column="time")
node = evset.node()

# Apply operators to create a processing graph.
sma = tp.simple_moving_average(node, window_length=tp.days(7))

# Run the graph on the input data.
result = sma.evaluate(evset)
```

## Key features

These are what set Temporian apart.

- **Simple and powerful API**: Temporian exports high level operations making processing complex programs short and ready to read.
- **Prevents modeling errors**: Temporian programs are guaranteed not to have future leakage unless the user calls the `leak` function, ensuring that models are not trained on future data.
- **Iterative development**: Temporian can be used to develop preprocessing pipelines in Colab or local notebooks, allowing users to visualize results each step of the way to identify and correct errors early on.
- **Efficient and well-tested implementations**: Temporian contains efficient and well-tested implementations of a variety of temporal data processing functions. For instance, our implementation of window operators is **x2000** faster than the same function implemented with NumPy.
- **Wide range of preprocessing functions**: Temporian contains a wide range of preprocessing functions, including moving window operations, lagging, calendar features, arithmetic operations, index manipulation and propagation, resampling, and more. For a full list of the available operators, see the [operators documentation](https://temporian.readthedocs.io/en/latest/reference/temporian/core/operators/).

## Documentation

The official documentation is available at [temporian.readthedocs.io](https://temporian.readthedocs.io/en/latest/).

## Contributing

Contributions to Temporian are welcome! Check out the [contributing guide](CONTRIBUTING.md) to get started.

## Credits

This project is a collaboration between Google and [Tryolabs](https://tryolabs.com/).
