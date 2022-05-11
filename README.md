# Prerequisites
1. Git
2. VS Code
   1. Remote Development Extension Pack (if using Docker)
3. Docker + Docker Compose (if using Docker)
4. Python >= 3.6 (if not using Docker)

# Quick start

## Using Docker
1. `git clone https://github.com/Minibrams/google-io-2022-rl.git`
2. `code google-io-2022-rl`
3. `Shift + CMD/CTRL + P -> Remote-Containers: Open Folder in Container`

## Not using Docker
1. `git clone https://github.com/Minibrams/google-io-2022-rl.git`
2. `code google-io-2022-rl`
3. `python3 -m venv .venv` (in VS Code terminal)
4. `source .venv/bin/activate` (in VS Code terminal)
5. `pip install -r requirements.txt`

# Train your model
To train your model, run:
```
python train.py
```

Modify `train.py` and `dqn/model.py` as you see fit.

When running the `train.py` script, your model is run (and trained) against a random agent.
Your model is saved to a file (`dqn_model.h5`) every 10 games.

# Play your model
To play the connect-four game against your model, run:
```
python game.py
```

If a `dqn_model.h5` file exists, the model is loaded before game start. Otherwise, you will be playing against a newly instantiated (random) agent.

# Test your model
To test your trained model against a random agent, run:
```
python test.py
```

The model will be loaded from `dqn_model.h5`.
The test will run 100 games against a random agent. The win rate will be reported at the end.