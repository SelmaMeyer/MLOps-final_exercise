import sys
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt 
import statistics

from data import mnist
from model import MyAwesomeModel
from torch import optim
from torch import nn

class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        train_set, _ = mnist()

        # make train loader
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

        # negative log likelihood
        criterion = nn.NLLLoss()

        # stochastic gradient descent
        optimizer = optim.Adam(model.parameters(), lr=0.003)

        epochs = 5

        epoch = []
        training_loss = []

        for e in range(epochs):
            running_loss = 0
            for images, labels in trainloader:

                # TRAINING

                # clear gradients
                optimizer.zero_grad()

                # use model to predict
                output = model(images)

                # calculate loss
                loss = criterion(output, labels)

                # backpropagate
                loss.backward()

                # take a gradient step/ optimize the weights to min loss
                optimizer.step()

                running_loss += loss.item()

            else:

                # Find the epoch and corresponding training loss for plotting
                epoch.append(e)
                training_loss.append(running_loss/len(trainloader))

                print(f"Training loss: {running_loss/len(trainloader)}")


        # Save model
        torch.save(model.state_dict(), 'checkpoint.pth')   

        # Plot the training loss for each epoch
        plt.plot(epoch, training_loss)    
        plt.show()

        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--load_model_from', default="checkpoint.pth")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        if args.load_model_from:

            checkpoint = torch.load(args.load_model_from)
            model = MyAwesomeModel()
            model.load_state_dict(checkpoint)

        _, test_set = mnist()

        testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

        running_accuracy = []

        with torch.no_grad():
            # set model to evaluation mode
            model.eval()
            
            for images, labels in testloader:
               
                ps = torch.exp(model(images))
        
                top_p, top_class = ps.topk(1, dim=1)
        
                equals = top_class == labels.view(*top_class.shape)
        
                accuracy = torch.mean(equals.type(torch.FloatTensor))

                running_accuracy.append(accuracy)
        
        print(f'Accuracy: {np.mean(running_accuracy)*100}%')
 

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    