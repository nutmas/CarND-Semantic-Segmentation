import matplotlib.pyplot as plt
import numpy as np

total_loss = [0.7, 0.3, 0.2, 0.19, 0.15, 0.14, 0.16, 0.13, 0.1]

plt.figure(figsize=[8,6])
plt.plot(total_loss, 'r', linewidth=2.0)
#plt.plot('loss value from log file', 'r', linewidth=3.0)
#plt.plot('val_loss from log file', 'b', linewidth=3.0)
plt.legend(['Training Loss', 'Validation Loss'], fontsize=18)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Loss', fontsize=16)
plt.yticks(np.arange(0, 1, step=0.05))
plt.grid()
plt.title('Loss Curves', fontsize=16)
plt.savefig('LossCurve.png', transparent=False, bbox_inches='tight')

plt.show()

plt.figure(figsize=[8,6])
plt.plot('acc value from log file', 'r', linewidth=3.0)
plt.plot('val_acc from log file', 'b', linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
plt.xlabel('Epochs', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title('Accuracy Curves', fontsize=16)

plt.show()