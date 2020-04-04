import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
import glob

resultsFiles = glob.glob("results/*.csv")

dfs = []
for fn in resultsFiles:
    df = pd.read_csv(fn)
    dfs.append(df)

df = pd.concat(dfs, axis=0)

# Filter out initial rows that don't contain match data
df = df.loc[df['elapsed_match'] != 0.0]
# Set multilevel index
df = df.set_index(['keypoint_type', 'descriptor_type'])

# Average over all matches for a particular combination of keypoint, descriptor
matchesFrame = df['num_matches_final'].groupby(['keypoint_type', 'descriptor_type']).mean().round(decimals=0)
kpTimeFrame = df.groupby('keypoint_type').mean().round(decimals=3)
dsTimeFrame = df.groupby('descriptor_type').mean().round(decimals=3)
matchTimeFrame = df.groupby('descriptor_type').mean().round(decimals=3)

print(kpTimeFrame)

# Save to csv
matchesFrame.to_csv('matching_results.csv')

# Plot
_, axes = plt.subplots(2, 2, figsize=(12, 12))
print(axes.shape)
sb.heatmap(matchesFrame.unstack(-1), annot=True, cmap='Blues', fmt='g', ax=axes[0, 0])
axes[0, 0].set_title("Number of Matches")

sb.barplot(x=kpTimeFrame.index, y=kpTimeFrame['elapsed_detect'], ax=axes[0, 1])
axes[0, 1].set_title("Runtime Perf - Detection")

sb.barplot(x=matchTimeFrame.index, y=matchTimeFrame['elapsed_match'], ax=axes[1, 0])
axes[1, 0].set_title("Runtime Perf - Matching Calculation")

sb.barplot(x=dsTimeFrame.index, y=dsTimeFrame['elapsed_desc_calc'], ax=axes[1, 1])
axes[1, 1].set_title("Runtime Perf - Descriptor Calculation")

plt.savefig("2d_feature_performance.png")
plt.show()
