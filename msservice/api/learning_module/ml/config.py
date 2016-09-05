class Config:
    reg_lambda = 0.01
    # MLP - Config
    max_epochs = 15
    batch_size = 100
    lr = 0.001
    n_inputs = 784
    n_classes = 10
    n_layers = 2
    hidden1_dim = 256
    hidden2_dim = 256
    model_path = './mlp/'
    display_step = 1
    device = ""
