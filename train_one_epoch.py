def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()  # Set the model to training mode
    losses = []  # List to store loss values

    # Iterate over batches of data in the data loader
    for step, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)  # Move images to the specified device
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]  # Move targets to the specified device

        loss_dict = model(images, targets)  # Compute the loss

        losses_dict = {k: v.item() for k, v in loss_dict.items()}  # Convert the loss tensor to Python float
        loss_value = sum(losses_dict.values())  # Sum the losses
        losses.append(loss_value)  # Append the sum of losses to the losses list

        # Print the loss value every print_freq steps
        if step % print_freq == 0:
            print(f"Epoch: {epoch}, Iteration: {step}, Loss: {loss_value}")

        optimizer.zero_grad()  # Reset the gradients
        sum(loss_dict.values()).backward()  # Compute the gradient
        optimizer.step()  # Update the model parameters

    # Return the average loss over the epoch
    return sum(losses) / len(losses)  