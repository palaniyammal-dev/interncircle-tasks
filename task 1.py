# =====================================================
# PROJECT: EXPLORATORY DATA ANALYSIS (EDA) ON TITANIC
# Libraries: Pandas, Matplotlib
# =====================================================

import pandas as pd
import matplotlib.pyplot as plt

# 1. LOAD DATASET
print("=" * 60)
print("TITANIC DATASET - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

df = pd.read_csv("titanic.csv")

print("\nDataset loaded successfully!")
print("Number of rows and columns:", df.shape)

# 2. DISPLAY FIRST 5 ROWS
print("\n--- FIRST 5 ROWS ---")
print(df.head())

# 3. DATASET INFORMATION
print("\n--- DATASET INFORMATION ---")
df.info()

# 4. STATISTICAL SUMMARY
print("\n--- STATISTICAL SUMMARY ---")
print(df.describe())

# 5. COLUMN NAMES
print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

# 6. CHECK MISSING VALUES
print("\n--- MISSING VALUES BEFORE CLEANING ---")
print(df.isnull().sum())

print("\nTotal missing values:", df.isnull().sum().sum())

# 7. CHECK DUPLICATES
print("\n--- DUPLICATE RECORDS ---")
print("Number of duplicate rows:", df.duplicated().sum())

# Remove exact duplicate rows
df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)

# 8. HANDLE MISSING VALUES

# Fill missing Age with median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked with mode
df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)

# Drop Cabin because many values are missing
df = df.drop(columns=["Cabin"])

# Fill any remaining missing numeric values with median
# and categorical values with mode
for column in df.columns:
    if df[column].isnull().sum() > 0:
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].median())
        else:
            df[column] = df[column].fillna(df[column].mode()[0])

print("\n--- MISSING VALUES AFTER CLEANING ---")
print(df.isnull().sum())

# 9. SURVIVAL ANALYSIS
print("\n--- SURVIVAL COUNTS ---")
print(df["Survived"].value_counts().sort_index())

survival_rate = df["Survived"].mean() * 100
print(f"\nOverall survival rate: {survival_rate:.2f}%")

# 10. GROUPING AND AGGREGATION

print("\n--- SURVIVAL RATE BY GENDER (%) ---")
gender_survival = df.groupby("Sex")["Survived"].mean() * 100
print(gender_survival.round(2))

print("\n--- SURVIVAL RATE BY PASSENGER CLASS (%) ---")
class_survival = df.groupby("Pclass")["Survived"].mean() * 100
print(class_survival.round(2))

print("\n--- AVERAGE AGE BY GENDER ---")
print(df.groupby("Sex")["Age"].mean().round(2))

print("\n--- AVERAGE FARE BY PASSENGER CLASS ---")
print(df.groupby("Pclass")["Fare"].mean().round(2))

print("\n--- MULTIPLE AGGREGATIONS BY CLASS ---")
summary = df.groupby("Pclass").agg(
    Passenger_Count=("PassengerId", "count"),
    Average_Age=("Age", "mean"),
    Average_Fare=("Fare", "mean"),
    Survival_Rate=("Survived", "mean")
)

summary["Survival_Rate"] = summary["Survival_Rate"] * 100
print(summary.round(2))

# 11. SURVIVAL ANALYSIS BY GENDER AND CLASS
print("\n--- SURVIVAL BY GENDER AND CLASS ---")
gender_class = df.groupby(["Sex", "Pclass"])["Survived"].agg(
    ["count", "sum", "mean"]
)

gender_class.columns = [
    "Total_Passengers",
    "Survivors",
    "Survival_Rate"
]

gender_class["Survival_Rate"] *= 100
print(gender_class.round(2))

# 12. VISUALIZATION 1: SURVIVAL COUNTS
survival_counts = df["Survived"].value_counts().sort_index()

plt.figure(figsize=(7, 5))
survival_counts.plot(kind="bar", color=["salmon", "seagreen"])

plt.title("Titanic Passenger Survival")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("survival_counts.png")
plt.show()

# 13. VISUALIZATION 2: SURVIVAL BY GENDER
gender_survival.plot(
    kind="bar",
    color=["steelblue", "coral"],
    figsize=(7, 5)
)

plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("survival_by_gender.png")
plt.show()

# 14. VISUALIZATION 3: SURVIVAL BY PASSENGER CLASS
class_survival.plot(
    kind="bar",
    color="orange",
    figsize=(7, 5)
)

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("survival_by_class.png")
plt.show()

# 15. VISUALIZATION 4: AGE DISTRIBUTION
plt.figure(figsize=(8, 5))
plt.hist(df["Age"], bins=20, color="skyblue", edgecolor="black")

plt.title("Age Distribution of Titanic Passengers")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("age_distribution.png")
plt.show()

# 16. SAVE CLEANED DATASET
df.to_csv("titanic_cleaned.csv", index=False)

print("\n--- CLEANED DATASET ---")
print("Cleaned dataset saved as titanic_cleaned.csv")

# 17. FINAL SUMMARY
print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)

print("Total passengers analyzed:", len(df))
print("Total columns after cleaning:", len(df.columns))
print("Missing values remaining:", df.isnull().sum().sum())
print("Duplicate rows remaining:", df.duplicated().sum())
print(f"Overall survival rate: {survival_rate:.2f}%")

print("\nKey Findings:")
print("1. Female passengers had a higher survival rate than male passengers.")
print("2. First-class passengers had a higher survival rate than third-class passengers.")
print("3. Missing values were handled using median, mode, and column removal.")
print("4. Grouping and aggregation helped identify survival patterns.")

print("\nAll EDA operations completed successfully!")

