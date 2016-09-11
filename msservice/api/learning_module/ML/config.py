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

class MNISTConfig:
    # MLP
    max_epochs = 15
    batch_size = 100
    lr = 0.001
    n_inputs = 784
    n_classes = 10
    n_layers = 2
    hidden1_dim = 256
    hidden2_dim = 256
    model_path = './mlp/model.ckpt'
    display_step = 1
    device = ""

class EventsMLPConfig:
    # MLP - Events
    max_epochs = 15
    batch_size = 100
    lr = 0.05
    n_inputs = 6
    n_classes = 2
    n_layers = 2
    hidden1_dim = 64
    hidden2_dim = 64
    model_path = './mlp/events/model.ckpt'
    display_step = 1
    device = ""

class CFMovieLensConfig:
    max_epochs = 60
    batch_size = 500
    lr = 0.20
    reg_lambda = 0.15
    m_users = 6040
    n_items = 3952
    dim = 64
    model_path = './cf/movielens/model.ckpt'
    device = "/cpu:0"
