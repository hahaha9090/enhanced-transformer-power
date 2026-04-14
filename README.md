# enhanced-transformer-power

Baseline codebase forked from the implementation of [A Transformer approach for Electricity Price Forecasting](https://arxiv.org/abs/2403.16108), intended as the starting point for subsequent development and version management.

## How to run the experiments

All the code is inside the `src` folder. To run the training for one model:

```bash
python -m src.train
```

Inside the main function of the train file, three variables manage the execution:

- `dataset`: to choose which dataset.
- `exec_mode`: to choose between training and evaluation of a model.
- `save_model`: to choose in evaluation if this model is saved as the best model for a specific dataset.

Then, to compare the results of the best models against the results of the models from the `epftoolbox`, run:

```bash
python -m src.benchmark
```

This benchmark has a variable in its main function called `benchmark` that manages the type of execution:

- `benchmark = dnn_last_year`: compute the results of the DNN normalizing with the last year's data.
- `benchmark = dnn_all_past`: compute the results of the DNN normalizing with all previous data.
- `benchmark = naive`: compute the results of a naive model that uses today's values as the forecast for the following day.
- `benchmark = results`: compute the final comparison between all models.

## Dependencies

Install the basic dependencies with:

```bash
pip install -r requirements.txt
```

Then install `epftoolbox`:

```bash
git clone https://github.com/jeslago/epftoolbox.git
cd epftoolbox
git checkout 7456ab84b42240b9c2519fb3b1bbbc52868a0817
pip install .
```

## Citation

Please cite the original paper if you find this baseline useful:

```bibtex
@misc{gonzalez2024transformer,
      title={A Transformer approach for Electricity Price Forecasting},
      author={Oscar Llorente Gonzalez and Jose Portela},
      year={2024},
      eprint={2403.16108},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
