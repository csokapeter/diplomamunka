import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('expert_act.csv', header=None)
df = df[df.iloc[:, -1] != 4]
df[df.columns[-1]] = df[df.columns[-1]].replace(5, 4)

df[0] = df[0] / 100
df[1] = df[1] / 2
df[2] = df[2] / 2
df[4] = df[4] / 7
df[5] = df[5] / 5
df[6] = df[6] / 4
df[11] = df[11] / 3
df.iloc[:, -6:-3] = df.iloc[:, -6:-3] / 7
df.iloc[:, -3:-1] = df.iloc[:, -3:-1] / 6
print(df.head())


counts = df.iloc[:, -1].value_counts().sort_index()
plt.bar(counts.index, counts.values, tick_label=counts.index)
plt.xlabel("Value")
plt.ylabel("Count")
plt.title("Distribution of expert actions")
plt.show()


# df.to_csv("expert_act_prep_5_action.csv", index=False, header=False)