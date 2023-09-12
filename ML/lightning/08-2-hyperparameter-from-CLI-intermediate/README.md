# Configure Hyperparameters from the CLI

## Requirements

```bash
pip install "lightning[pytorch-extra]"
```

or if only interested in `LightningCLI`, just install jsonargparse:

```bash
pip install "jsonargparse[signatures]"
```

Now your model can be managed via the CLI. To see the available commands type:

```bash
$ python main.py --help
$ python main.py fit
$ python main.py validate
$ python main.py test
$ python main.py predict

```