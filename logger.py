

class Logger:
    def __init__(self, labelString):
        self.labelString = labelString

    def log(self, now, loss, accuracy, i):
        print("%d-%d-%d %2d:%2d:%2d: Step %6d: %s accuracy: %.5g \t\t loss: %.5g" % (now.month, now.day, now.year, now.hour, now.minute, now.second, i, self.labelString, accuracy, loss ))

