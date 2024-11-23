import torch

def save(path, model, settings, optimizer=None, epoch_train_losses=None, epoch_val_losses=None, inference=True):
    """
    Saves a PyTorch model and its settings to the specified path. Optionally saves the optimizer, training/validation 
    losses if not in inference mode.

    Parameters:
    model (torch.nn.Module): The PyTorch model to be saved.
    optimizer (torch.optim.Optimizer): The optimizer used for training the model.
    path (str): The file path where the model, optimizer, losses, and settings should be saved.
    epoch_train_losses (list): List of training losses for each epoch.
    epoch_val_losses (list): List of validation losses for each epoch.
    settings (dict): The settings dictionary containing model and training configurations.
    inference (bool): If True, save only the model's state dictionary and settings for inference purposes.
    """
    save_content = {
        'model_state_dict': model.state_dict(),
        'settings': settings
    }

    if not inference:
        save_content['optimizer_state_dict'] = optimizer.state_dict()
        save_content['epoch_train_losses'] = epoch_train_losses
        save_content['epoch_val_losses'] = epoch_val_losses

    torch.save(save_content, path)

    if inference:
        print(f"Inference model saved to {path}")
    else:
        print(f"Training model saved to {path}")


def load(path, model_class, device=None, optimizer_class=None, inference=False):
    """
    Loads a PyTorch model and its settings from the specified path. Optionally loads the optimizer, training/validation 
    losses if not in inference mode.

    Parameters:
    path (str): The file path where the model, optimizer, losses, and settings are saved.
    model_class (class): The class of the model to be loaded.
    optimizer_class (class): The class of the optimizer to be used with the model (if not in inference mode).
    inference (bool): If True, load only the model's state dictionary and settings for inference purposes.

    Returns:
    model (torch.nn.Module): The loaded PyTorch model.
    Optional returns if not in inference mode:
    optimizer (torch.optim.Optimizer): The optimizer for the model.
    epoch_train_losses (list): List of training losses for each epoch.
    epoch_val_losses (list): List of validation losses for each epoch.
    settings (dict): The settings dictionary containing model and training configurations.
    """
    checkpoint = torch.load(path)
    settings = checkpoint['settings']
    if device:
        settings['model']['device'] = device

    model = model_class(**settings['model'])
    model.load_state_dict(checkpoint['model_state_dict'])

    if not inference and optimizer_class is not None:
        optimizer = optimizer_class(model.parameters(), lr = settings['train']['lr'], weight_decay = settings['train']['l2'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_train_losses = checkpoint.get('epoch_train_losses', [])
        epoch_val_losses = checkpoint.get('epoch_val_losses', [])
        print(f"Training model loaded from {path}")
        return model, settings, optimizer, epoch_train_losses, epoch_val_losses

    print(f"Inference model loaded from {path}")
    return model, settings
