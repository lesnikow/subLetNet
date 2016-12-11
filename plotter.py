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
    plt.grid(True)
    global alpha, trail, time_0
    alpha = 0.250
    trail = 15
    time_0 = time.time()
    
    def plotTrainVsTestAcc(self, directory, batchsz):
        """Plots train loss vs test accuracy """
        plt.clf()
        plt.title('Regression net on MNIST')
        time_delta = repr('%.5g' % (time.time() - time_0) )
        plt.xlabel('batches of size ' + str(batchsz) + ', training time: ' + str(time_delta) + 'secs')
        plt.ylabel('accuracy')
        
        plt.plot(self.x_axis_acc_train, self.acc_train, label='train acc', alpha=alpha)
        plt.plot(self.x_axis_acc_test, self.acc_test, label='test acc', alpha=alpha)

        acc_train = self.acc_train
        trail_train = [sum(acc_train[i-trail: i]) / float(trail) for i in range(len(acc_train))]
        if len(self.x_axis_acc_train) >= trail:
            plt.plot(self.x_axis_acc_train[trail:], trail_train[trail:], label='train acc trailing average', alpha=1.0)
        
        acc_test = self.acc_test
        trail_test = [sum(acc_test[i-trail: i]) / float(trail) for i in range(len(acc_test))]
        if len(self.x_axis_acc_test) >= trail:
            plt.plot(self.x_axis_acc_test[trail:], trail_test[trail:], label='test acc trailing average', alpha=1.0)

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
        
    def plot(self, batches, perplexities):
        plt.plot(batches, perplexities)
        
    def savefig(self, dir):
        plt.savefig(dir + self.dateString + '.png')
       
    def plotSave(self, x_axis, losses, directory):
        plt.plot(x_axis, losses)
        plt.savefig(directory + self.dateString + '.png')

