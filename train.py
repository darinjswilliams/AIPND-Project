import torch

from makeparse import make_parser
from model_utils import (get_datasets,
                         get_model_architecture, get_dataloaders,
                         setup_hyper_params)
from predict_utils import save_checkpoint
from validation_utilis import get_device, commandline_validations


def train_model(
                trainloaders,
                validloaders,
                model,
                criterion,
                optimizer,
                epochs
               ):

    # move the model to the device
    device = get_device()
    model.to(device)

    # set up paramter that are used to train our model
    epochs = epochs
    running_loss = 0
    steps = 0
    print_everything = 10
    print(model, trainloaders, validloaders, criterion, optimizer, epochs, running_loss, steps, print_everything)
    print("Training Started")
    for epoch in range(epochs):

        for images, labels in trainloaders:

            # Accumulate Steps
            steps += 1

            # Move input and label tensors to the GPU or Device
            images, labels = images.to(device), labels.to(device)

            # Zero out gradient
            optimizer.zero_grad()

            # Get log probabilities
            logps = model.forward(images)

            # Get Los from criterion
            loss = criterion(logps, labels)

            # Do backward pass
            loss.backward()

            # Take a step
            optimizer.step()

            # increment running loss
            running_loss += loss.item()

            # Drop out of the training loop and test our network accuracy on test data set

            if steps % print_everything == 0:

                # set our model to evaluation inference mode to turn off dropout
                model.eval()

                valid_loss = 0
                accuracy = 0
                with torch.no_grad():
                    for images, labels in validloaders:
                        # Move input and label tensors to the GPU or Device
                        images, labels = images.to(device), labels.to(device)

                        # Get log probabilities
                        logps = model(images)

                        # Get Loss from criterion
                        loss = criterion(logps, labels)

                        # Accumulate valid loss
                        valid_loss += loss.item()

                        # calculate accuracy, remember out model is returning logsoftmax
                        # so its the log probabilities of our classes to get actual probabilites
                        ps = torch.exp(logps)

                        # Get the top probabilites and class
                        # Providing 1, will give you the first largest value in your probability
                        # Make sure you set the dim = 1, so it will actually look for the top probability
                        # along the columns
                        top_ps, top_class = ps.topk(1, dim=1)

                        # Check for equality against your labels with the equality tensor
                        equality = top_class == labels.view(*top_class.shape)

                        # Update accuracy, remember to use equality, once you change it to FloatTensor
                        # than you can use torch.mean
                        accuracy += torch.mean(equality.type(torch.FloatTensor))

                    # Print Information
                    print('Epoch: {}/{}'.format(epoch + 1, epochs),
                          "Training Loss: {:.3f}...".format(running_loss / print_everything),
                          "Validation Loss: {:.3f}...".format(valid_loss / len(validloaders)),
                          'Validation Accuracy {:.3f}%'.format(accuracy / len(validloaders)))

                    # Set running loss back to 0
                    runninng_loss = 0

                    # Set model back to Training
                    model.train()
    print("Training Completed")

def test_model(model, testloaders, criterion):

    model.eval()

    # Accumulate loss and Accuracy
    test_loss = 0
    accuracy = 0

    #move to device
    device = get_device()
    print("Testing Started")
    # No Gradient - Turn off AutoGrad
    with torch.no_grad():
        # Load Data from TestLoader
        for images, labels in testloaders:
            # Move input and label tensors to the GPU or Device
            images, labels = images.to(device), labels.to(device)

            # Get log probabilities
            logps = model.forward(images)

            # Get Loss from criterion
            loss = criterion(logps, labels)

            # Accumulate valid loss
            test_loss += loss.item()

            # calculate accuracy, remember out model is returning logsoftmax
            ps = torch.exp(logps)

            # Get the top probabilites and class
            top_ps, top_class = ps.topk(1, dim=1)

            # Check for equality against your labels with the equality tensor
            equality = top_class == labels.view(*top_class.shape)

            # Update accuracy, remember to use equality, once you change it to FloatTensor
            accuracy += torch.mean(equality.type(torch.FloatTensor))

        print('Testing Loss: {:.3f}...'.format(test_loss / len(testloaders)),
              'Testting Accuracy {:.3f}%'.format(accuracy / len(testloaders)))

    # Set model back to Training
    model.train()
    print("Testing Completed")



def main():

    input_args = make_parser()
    file_path = commandline_validations(input_args)
    train_dataset, valid_dataset, test_dataset, image_dataset = get_datasets(file_path)
    train_loader, valid_loader, test_loader \
        = get_dataloaders(train_dataset, valid_dataset, test_dataset)
    model = get_model_architecture(input_args)

    # Convert pathlib path to string
    training_type = input_args.dir
    # Pass model, model name, hidden units, learning rate

    model, criterion, optimizer = setup_hyper_params( model,
                                            input_args.arch,
                                            input_args.hidden_units,
                                            input_args.learning_rate)

    if input_args.dir in ['train', 'valid']:
        train_model(trainloaders=train_loader,
                            validloaders=valid_loader,
                            model=model,
                            criterion=criterion,
                            optimizer=optimizer,
                            epochs=input_args.epochs
                            )
    elif training_type == 'test':
        test_model(testloaders=train_loader,
                       model=model,
                       criterion=criterion,
                       )

    # Lets Save the Model Parameter for Later Use for Inference
    # Defaults to Checkpoint
    # When user uses predict the model will load
    save_checkpoint(image_datasets=image_dataset,
                    model=model,
                    classifier=model.classifier,
                    optimizer=optimizer,
                    epochs=input_args.epochs,
                    input_features=model.classifier[0].in_features,
                    hidden_layer=input_args.hidden_units,
                    learning_rate=input_args.learning_rate,
                    path=input_args.checkpoint)

if __name__ == '__main__':
    # main()
    input_args = make_parser()
    print(input_args)
    print(input_args.arch)
    file_path = commandline_validations(input_args)
    train_datasets, valid_datasets, test_datasets, image_datasets = get_datasets(file_path)
    # # print(get_datasets(file_path))
    # train_loader, valid_loader, test_loader = get_process_path(file_path)
    model = get_model_architecture(input_args)
    model, criterion, optimizer = setup_hyper_params(model,
                                                     input_args.arch,
                                                     input_args.hidden_units,
                                                     input_args.learning_rate)
    save_checkpoint(image_datasets=image_datasets,
                    model=model,
                    classifier=model.classifier,
                    optimizer=optimizer,
                    epochs=input_args.epochs,
                    input_features=model.classifier[0].in_features,
                    hidden_layer=input_args.hidden_units,
                    learning_rate=input_args.learning_rate,
                    file_path=input_args)











