import torch
from .config import DEVICE, IOU_THRESHOLD
from .metrics import iou_metric

from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast

def train_model(
        model, 
        train_dataloader, 
        epoches, 
        optimizer, 
        lr_scheduler, 
        valid_dataloader=None, 
        early_stopping=None, 
        model_checkpoint=None):
    """
        Train model
    """

    def train(model, train_dataloader, epoch, optimizer, scaler):
        model.train()
        total_loss = 0
        total_samples = 0

        # Use tqdm for progress bar
        with tqdm(train_dataloader, desc=f"Epoch {epoch}/{epoches}") as progress_bar:
            for images, targets in progress_bar:
                images = [image.to(DEVICE) for image in images]
                targets = [{k: v.to(DEVICE) for k, v in target.items()} for target in targets]
                
                optimizer.zero_grad()

                with autocast():
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                scaler.scale(losses).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += losses.item() * len(images)
                total_samples += len(images)
                
                # Update the tqdm progress bar description
                progress_bar.set_postfix(loss=(total_loss / total_samples), refresh=True)

        return total_loss / total_samples


    def validate(model, valid_dataloader):
        total_loss = 0
        total_samples = 0
        iou_scores = []
        for images, targets in valid_dataloader:
            images = [image.to(DEVICE) for image in images]
            targets = [{k: v.to(DEVICE) for k, v in target.items()} for target in targets]
            # Temporarily switch to training mode to calculate validation losses
            model.train() 
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item() * len(images)
            total_samples += len(images)

            # Switch back to evaluation mode to get predictions
            model.eval()
            with torch.no_grad():
                predictions = model(images)

            # Compare between predictions and ground truth
            for pred, target in zip(predictions, targets):
                pred_boxes = pred["boxes"].cpu()
                target_boxes = target["boxes"].cpu()
                for pred_box in pred_boxes:
                    best_iou = 0.0
                    for true_box in target_boxes:
                        iou = iou_metric(pred_box, true_box)
                        if iou > best_iou:
                            best_iou = iou
                    iou_scores.append(best_iou)

        # Calculate average validation loss and accuracy
        validation_loss = total_loss / total_samples
        corrects = torch.sum(torch.tensor(iou_scores) > IOU_THRESHOLD).item()
        validation_accuracy = 100.0 * (corrects / len(iou_scores))
        
        print(f'Validation set: Average loss: {validation_loss:.4f} - Accuracy: {corrects}/{len(iou_scores)} ({validation_accuracy:.2f}%)\n')
        return validation_loss, validation_accuracy

    # Train model
    history = {
        'loss': [],
    }
    if valid_dataloader:
        history['val_loss'] = []
        history['val_accuracy'] = []

    scaler = GradScaler()

    # Train model    
    model.to(DEVICE)
    for epoch in range(1, epoches + 1):
        loss = train(model, train_dataloader, epoch, optimizer, scaler)
        history['loss'].append(loss)
        if valid_dataloader:
            val_loss, val_accuracy = validate(model, valid_dataloader)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
        if not early_stopping is None:
            early_stopping(model, val_loss or loss)
        if not model_checkpoint is None:
            model_checkpoint(model, val_loss or loss, epoch, optimizer)
        lr_scheduler.step()
        
    return history