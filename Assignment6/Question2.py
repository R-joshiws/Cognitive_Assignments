import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = {
    "Subject": ["Math", "Science", "English", "History", "Computer"],
    "Score": [85, 92, 78, 88, 95]
}

df = pd.DataFrame(data)

sns.set(style="whitegrid")

colors = sns.color_palette("husl", len(df))
plt.figure(figsize=(8, 5))
ax = sns.barplot(x="Subject", y="Score", data=df, palette=colors)

for i, score in enumerate(df["Score"]):
    ax.text(i, score + 1, str(score), ha='center', fontsize=12, fontweight='bold')

title = "Student Scores in Different Subjects"
plt.title(title, fontsize=14, fontweight='bold')
plt.xlabel("Subjects", fontsize=12)
plt.ylabel("Scores", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()
