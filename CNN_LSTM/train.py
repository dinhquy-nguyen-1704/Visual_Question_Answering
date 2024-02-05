from src.evaluate import evaluate

def fit(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        epochs
):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        batch_train_losses = []
        model.train()
        for idx, (images, questions, labels) in enumerate(train_loader):
            images, questions, labels = images.to(device), questions.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, questions)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_train_losses.append(loss.item())

        train_loss = sum(batch_train_losses) / len(batch_train_losses)
        train_losses.append(train_loss)

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        val_losses.append(val_loss)

        print(f'EPOCH {epoch + 1}: Train loss: {train_loss:.4f} Val loss: {val_loss:.4f} Val Acc:{val_acc:.4f}')

        scheduler.step()

    return train_losses, val_losses, model
