import torch
from torch_geometric.data import Data
from tqdm import tqdm
from .....config import log_config

TRAIN_LOSS_TEXT = "Train loss"
TEST_LOSS_TEXT = "Test loss"


class ModelWrapper:
    """ModelWrapper class.

    """
    def __init__(self, n_features, n_targets, model_class: type[torch.nn.Module], path: str, *args, **kwargs):
        """Initialize the instance.

        Parameters
        ----------
        n_features : Any
            Description.
        n_targets : Any
            Description.
        model_class : type[torch.nn.Module]
            Description.
        path : str
            Description.
        *args : tuple
            Description.
        **kwargs : dict
            Description.

        Returns
        -------
        None
            Description.

        """
        self.n_features = n_features
        self.n_targets = n_targets
        self.model = model_class(n_features, n_targets, *args, **kwargs)
        self.load_model(path)

    def load_model(self, file_path: str):
        """Load model.

        Parameters
        ----------
        file_path : str
            Description.

        """
        state_dict = torch.load(file_path, weights_only=True)
        self.model.load_state_dict(state_dict)

    def save_model(self, file_path: str):
        """Save model.

        Parameters
        ----------
        file_path : str
            Description.

        """
        state_dict = self.model.state_dict()
        torch.save(state_dict, file_path)

    def _train_model(self, data: Data, epochs: int, learning_rate: float, weight_decay: float, loss_fn, **kwargs):

        """Train model.

        Parameters
        ----------
        data : Data
            Description.
        epochs : int
            Description.
        learning_rate : float
            Description.
        weight_decay : float
            Description.
        loss_fn : Any
            Description.
        **kwargs : dict
            Description.

        """
        model = self.model
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        pbar = tqdm(
            range(epochs), disable=log_config.disable_tqdm, desc=f"{TRAIN_LOSS_TEXT}: ...... | {TEST_LOSS_TEXT}: ......"
        )

        train_losses = []
        test_losses = []

        for _ in pbar:
            model.train()
            optimizer.zero_grad()
            out = model(data)
            train_loss = loss_fn(out[data.train_mask], data.y[data.train_mask], **kwargs)
            train_loss.backward()
            optimizer.step()

            train_loss = train_loss.item()
            train_losses.append(train_loss)

            # Validation phase
            test_loss = self._test_model(data, loss_fn, **kwargs)
            test_losses.append(test_loss)

            pbar.set_description(f"{TRAIN_LOSS_TEXT}: {train_loss:.5f} | {TEST_LOSS_TEXT}: {test_loss:.5f}")

        return train_losses, test_losses

    def _evaluate_model(self, data: Data):
        """Evaluate model.

        Parameters
        ----------
        data : Data
            Description.

        """
        model = self.model
        model.eval()
        with torch.no_grad():
            out = model(data)
        return out

    def _test_model(self, data: Data, loss_fn, **kwargs):
        """Test model.

        Parameters
        ----------
        data : Data
            Description.
        loss_fn : Any
            Description.
        **kwargs : dict
            Description.

        """
        out = self._evaluate_model(data)
        loss = loss_fn(out[data.test_mask], data.y[data.test_mask], **kwargs)
        return loss.item()
