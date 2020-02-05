import numpy as np
import matplotlib.pyplot as plt
import pandas

# import some data


# plot data
fig = plt.figure()
ax  = fig.add_subplot(111)
ax.plot(y, c='b', label=label[0])
ax.plot(y, c='r', label=label[1])

leg = plt.legend()
# get the lines and texts inside legend box
leg_lines = leg.get_lines()
leg_texts = leg.get_texts()
# bulk-set the properties of all lines and texts
plt.setp(leg_lines, linewidth=4)
plt.setp(leg_texts, fontsize='x-large')
plt.savefig('leg_example')
plt.show()
