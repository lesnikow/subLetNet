"""
Object to plot training and test losses
"""
import matplotlib.pyplot as plt
import time

class Plotter:
    dateString = time.strftime("%d_%m_%Y@%H:%M:%S")
    x_axis_losses_train, x_axis_losses_test, losses_train, losses_test = [], [], [], []
    x_axis_acc_train, x_axis_acc_test, acc_train, acc_test = [], [], [], []
    plt.rcParams['legend.loc'] = 'best'
    
    def plot(self, batches, perplexities):
        plt.plot(batches, perplexities)
        
    def savefig(self, dir):
        plt.savefig(dir + self.dateString + '.png')
       
    def plotSave(self, x_axis, losses, directory):
        plt.plot(x_axis, losses)
        plt.savefig(directory + self.dateString + '.png')
    
    def plotTrainVsTestAcc(self, directory):
        """Plots train loss vs test accuracy """
        plt.clf()
        plt.plot(self.x_axis_acc_train, self.acc_train, label='train acc')
        plt.plot(self.x_axis_acc_test, self.acc_test, label='test acc')
        plt.legend(framealpha=0.5)
        plt.savefig(directory + self.dateString + 'TrainVsTestAcc.png')
   
    def plotTrainVsTestLosses(self, directory):
        """Plots train loss vs test loss """
        plt.clf()
        plt.plot(self.x_axis_losses_train, self.losses_train, label='train loss')
        plt.plot(self.x_axis_losses_test, self.losses_test, label='test loss')
        plt.legend(framealpha=0.5)
        plt.savefig(directory + self.dateString + 'TrainVsTestLoss.png')
    
    def plotTrainLossAccuracy(self, directory):
        """Plots both the accuracy and cross ent loss and saves as separate figures. """
        plt.plot(self.x_axis_losses_train, self.losses_train)
        plt.savefig(directory + self.dateString + 'TrainCrosEntLoss.png')
        plt.clf()
        plt.plot(self.x_axis_acc_train, self.accuracy_train)
        plt.savefig(directory + self.dateString + 'TrainAccuracy.png')

        
