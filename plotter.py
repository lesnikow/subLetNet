"""
Object to plot training and test losses
"""
import matplotlib.pyplot as plt
import time

class Plotter:
    dateString = time.strftime("%d_%m_%Y@%H:%M:%S")
    x_axis_train, x_axis_test, losses_train, losses_test = [], [], [], []
    x_axis_acc_train, x_axis_acc_test, accuracy_train, accuracy_test = [], [], [], []
    plt.rcParams['legend.loc'] = 'best'
    
    def plot(self, batches, perplexities):
        plt.plot(batches, perplexities)
        
    def savefig(self, dir):
        plt.savefig(dir + self.dateString + '.png')
       
    def plotSave(self, x_axis, losses, directory):
        plt.plot(x_axis, losses)
        plt.savefig(directory + self.dateString + '.png')
   
    def plotSave(self, x_axis_train, x_axis_test, losses_train, losses_test, directory):
        plt.clf()
        plt.plot(x_axis_train, losses_train, label='train loss')
        plt.plot(x_axis_test, losses_test, label='test loss')
        plt.legend(framealpha=0.5)
        plt.savefig(directory + self.dateString + '.png')
    
    def plotSaveTrain(self, directory):
        plt.plot(self.x_axis_train, self.losses_train)
        plt.savefig(directory + self.dateString + 'CrosEntLoss.png')
        plt.clf()
        plt.plot(self.x_axis_acc_train, self.accuracy_train)
        plt.savefig(directory + self.dateString + 'Accuracy.png')

        
