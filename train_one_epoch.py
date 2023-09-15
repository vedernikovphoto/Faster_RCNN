def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    """
    Trains the model for one epoch and returns the average loss.

    Args:
        model: The model to be trained.
        optimizer: The optimizer for training the model.
        data_loader: DataLoader providing the training data.
        device: The device on which to perform the computation (CPU or GPU).
        epoch: The current epoch number.
        print_freq: Frequency at which to print the loss values.

    Returns:
        float: The average loss over the epoch.
    """

    model.train()
    losses = []

    for step, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)  # Move images to the specified device
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # Move targets to the specified device

        loss_dict = model(images, targets)

        losses_dict = {k: v.item() for k, v in loss_dict.items()}  # Convert the loss tensor to Python float
        loss_value = sum(losses_dict.values())
        losses.append(loss_value)

        if step % print_freq == 0:
            print(f"Epoch: {epoch}, Iteration: {step}, Loss: {loss_value}")

        optimizer.zero_grad()
        sum(loss_dict.values()).backward()
        optimizer.step()

    return sum(losses) / len(losses)
