class Plots:
    def abc(self, loss, val_loss, accuracy, val_accuracy):
        import matplotlib.pylab as plt
        plt.figure()
        plt.ylabel("Loss (training and validation)")
        plt.xlabel("Training Steps")
        plt.ylim([0, 2])
        plt.plot(loss)
        plt.plot(val_loss)

        plt.figure()
        plt.ylabel("Accuracy (training and validation)")
        plt.xlabel("Training Steps")
        plt.ylim([0, 1])
        plt.plot(accuracy)
        plt.plot(val_accuracy)
