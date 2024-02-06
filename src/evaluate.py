import torch

def evaluate(model, dataloader, criterion):

    model.eval()
    correct = 0
    total = 0
    losses = []

    with torch.no_grad():
        for idx, inputs in enumerate(dataloader):

            images = inputs['image']
            questions = inputs['question']
            labels = inputs['label']

            outputs = model(images, questions)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss = sum(losses) / len(losses)
    acc = correct / total

    return loss, acc
