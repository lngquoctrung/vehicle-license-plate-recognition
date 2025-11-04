import torch
import os

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0):
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
    def __init__(self, filepath, save_best_only=False, best_checkpoint_filepath=None):
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.best_loss = float("inf")
        self.best_checkpoint_filepath = best_checkpoint_filepath

    def __call__(self, model, val_loss, epoch, optimizer, history):
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "best_loss": val_loss,
            "optimizer": optimizer.state_dict(), 
            "history": history
        }
        # Create a checkpoint directory if it doesn't exist
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

        # Always save the latest checkpoint
        torch.save(checkpoint, self.filepath)
        print(f"Latest checkpoint saved at {self.filepath} \n")

        # Save best checkpoint
        if self.save_best_only and val_loss < self.best_loss and self.best_checkpoint_filepath:
            os.makedirs(os.path.dirname(self.best_checkpoint_filepath), exist_ok=True)
            self.best_loss = val_loss
            torch.save(checkpoint, self.best_checkpoint_filepath)
            print(f"Best model saved with loss {val_loss:.4f}")