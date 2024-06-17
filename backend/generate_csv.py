import os
import pandas as pd

filenames = []
labels = []

for filename in os.listdir('ants'):
    filenames.append(filename)
    # labels.append(0)

for filename in os.listdir('bees'):
    filenames.append(filename)
    # labels.append(1)

data = {
    'filename': filenames,
    # 'label': labels
}

df = pd.DataFrame(data)
df.to_csv('unlabeled.csv', index=False)
