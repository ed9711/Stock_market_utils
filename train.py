import time
import torch
import copy


def train_model(model, criterion, optimizer, scheduler, trainLoad, validLoad, device, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    last_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    worst_acc = 1

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                d = trainLoad
                x = 0
            else:
                model.eval()
                x = 1
                d = validLoad # Set model to evaluate mode
            dataset_sizes = [len(trainLoad), len(validLoad)]
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for batch, (inputs, labels) in enumerate(d):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs.unsqueeze(1))
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels.T[0])

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics

                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)

                if preds[0] == labels.data[0]:
                    running_corrects+=1
                if preds[1] == labels.data[1]:
                    running_corrects+=1

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / (dataset_sizes[x] * len(labels))
            # epoch_acc = running_corrects.double() / dataset_sizes[x]
            epoch_acc = running_corrects / (dataset_sizes[x] * len(labels))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            # if phase == 'val' and epoch_acc < worst_acc:
            #     worst_acc = epoch_acc
            #     worst_model_wts = copy.deepcopy(model.state_dict())
            last_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Worst val Acc: {:4f}'.format(worst_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, last_model_wts
