import torch
import os

from .config import CHECKPOINT_DIR, FINAL_STATE_DIR
from .utils import create_directory

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.best_score = None
        self.early_stopping = False
        self.best_model_state = None
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0

    def __call__(self, model, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
            self.best_model_state = model.state_dict()
        elif val_loss > self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stopping = True
        else:
            self.best_score = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict()

class ModelCheckpoints:
    def __init__(self, filename, save_best_only=True):
        self.filename = filename
        self.save_best_only = save_best_only

    def __call__(self, model, val_loss, epoch, optimizer):
        checkpoint = {
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "best_loss": val_loss,
            "optimizer": optimizer.state_dict(),
        }
        create_directory(CHECKPOINT_DIR)
        file_path = os.path.join(CHECKPOINT_DIR, self.filename)
        torch.save(checkpoint, file_path)

        if self.save_best_only:
            create_directory(FINAL_STATE_DIR)
            best_model_file_path = os.path.join(FINAL_STATE_DIR, f"best_{self.filename}")
            torch.save(checkpoint, best_model_file_path)