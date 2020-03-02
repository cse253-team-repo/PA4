# CSE 253 Project 4



### Configuration file

The configuration files are stored in the directory `config/config.yaml`. To utilize the model pass the correct configuration file to the scripts. The specified hyperparameters are specialized for training process. 

### Train the model

To train the model, pass correct yaml file to the scripts and run the following scripts.

```bash
python3 train_baseline.py
```

We design a vanilla RNN and a LSTM decoder to reconstruct the text caption form corresponding images

### Model evaluation

```
python test.py
```

It loads the specified model path and generate the text description. A temperature score
