import torch
import os
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from .config import Config
from .metrics import iou_metric, accuracy_metric

def train_model(
    model,
    train_dataloader,
    epoches,
    optimizer,
    lr_scheduler,
    valid_dataloader=None,
    early_stopping=None,
    model_checkpoint=None,
    resume_from_checkpoint=None):
    """
    Train model
    """
    def train(model, train_dataloader, epoch, optimizer, scaler):
        model.train()
        total_loss = 0
        total_samples = 0
        with tqdm(train_dataloader, desc=f"Epoch {epoch}/{epoches}") as progress_bar:
            for images, targets in progress_bar:
                if images is None or targets is None or len(images) == 0:
                    continue
                images = [image.to(Config.DEVICE) for image in images]
                targets = [{k: v.to(Config.DEVICE) for k, v in target.items()} for target in targets]
                optimizer.zero_grad()
                with autocast(device_type=str(Config.DEVICE).split(':')[0]):
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += losses.item() * len(images)
                total_samples += len(images)
                progress_bar.set_postfix({
                    'loss': total_loss / total_samples
                })
        return total_loss / total_samples

    def validate(model, valid_dataloader):
        model.eval()
        total_loss = 0
        total_iou = 0
        total_acc = 0
        total_samples = 0
        with torch.no_grad():
            for images, targets in valid_dataloader:
                if images is None or targets is None or len(images) == 0:
                    continue
                images = [image.to(Config.DEVICE) for image in images]
                targets = [{k: v.to(Config.DEVICE) for k, v in target.items()} for target in targets]
                
                # Calculate loss - TEMPORARILY SET TO TRAIN MODE
                model.train()
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                model.eval()
                
                # Calculate IoU and Accuracy
                predictions = model(images)
                batch_iou = 0
                batch_acc = 0
                for pred, target in zip(predictions, targets):
                    pred_boxes = pred['boxes'].cpu()
                    target_boxes = target['boxes'].cpu()
                    if len(pred_boxes) > 0 and len(target_boxes) > 0:
                        # Calculate IoU
                        for pred_box in pred_boxes:
                            best_iou = 0.0
                            for true_box in target_boxes:
                                iou = iou_metric(pred_box, true_box)
                                best_iou = max(best_iou, iou)
                            batch_iou += best_iou
                        batch_iou /= len(pred_boxes)
                        # Calculate Accuracy
                        acc = accuracy_metric(pred_boxes, target_boxes, Config.IOU_THRESHOLD)
                        batch_acc += acc
                batch_iou /= len(images) if len(images) > 0 else 1
                batch_acc /= len(images) if len(images) > 0 else 1
                total_loss += losses.item() * len(images)
                total_iou += batch_iou * len(images)
                total_acc += batch_acc * len(images)
                total_samples += len(images)
        return (total_loss / total_samples,
                total_iou / total_samples,
                total_acc / total_samples)

    # Initialize history - FIX: Add closing brace
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_iou': [],
        'val_acc': []
    }

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()

    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_from_checkpoint is not None and os.path.exists(resume_from_checkpoint):
        checkpoint = torch.load(resume_from_checkpoint, map_location=Config.DEVICE)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        history = checkpoint.get('history', history)
        print(f"Resumed training from epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, epoches):
        # Training
        train_loss = train(model, train_dataloader, epoch, optimizer, scaler)
        history['train_loss'].append(train_loss)
        print(f"Epoch {epoch}/{epoches} - Train Loss: {train_loss:.4f}")

        # Validation
        if valid_dataloader is not None:
            val_loss, val_iou, val_acc = validate(model, valid_dataloader)
            history['val_loss'].append(val_loss)
            history['val_iou'].append(val_iou)
            history['val_acc'].append(val_acc)
            print(f"Epoch {epoch}/{epoches} - Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}, Val Acc: {val_acc:.4f}")

            # Early stopping
            if early_stopping is not None:
                early_stopping(model, val_loss)
                if early_stopping.early_stopping:
                    print(f"Early stopping triggered at epoch {epoch}")
                    model.load_state_dict(early_stopping.best_model_state)
                    break

            # Model checkpoint
            if model_checkpoint is not None:
                model_checkpoint(model, val_loss, epoch, optimizer, history)

        # Learning rate scheduler
        lr_scheduler.step()

    return history