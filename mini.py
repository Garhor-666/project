import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from statsmodels.formula.api import ols

# Data Analysis class from gex2.py (ANOVA)
class DataAnalysis:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully. {self.df.shape[0]} rows and {self.df.shape[1]} columns.")
        self.column_types = self.list_column_types()

    def list_column_types(self):
        """Lists all column types, correctly classifying ordinal and nominal."""
        column_types = {
            'interval': [],
            'numeric_ordinal': [],
            'non_numeric_ordinal': [],
            'nominal': []
        }
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                unique_values = self.df[col].nunique()
                if unique_values > 5:
                    column_types['interval'].append(col)
                else:
                    column_types['numeric_ordinal'].append(col)
            elif pd.api.types.is_object_dtype(self.df[col]):
                column_types['nominal'].append(col)

        print("\nClassified variables:")
        for data_type, cols in column_types.items():
            print(f"\n{data_type.upper()} Variables:")
            for col in cols:
                print(f"- {col} ({self.df[col].nunique()} unique values)")

        return column_types

    def plot_distribution(self, variable):
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df[variable], kde=True)
        plt.title(f'Distribution plot for {variable}')
        plt.xlabel(variable)
        plt.ylabel('Frequency')
        plt.grid()
        plt.show()

    def check_normality(self, column):
        """Check normality using the Shapiro-Wilk test."""
        data = self.df[column].dropna()
        stat, p_value = stats.shapiro(data)
        print(f"Shapiro-Wilk Test: Statistic = {stat:.4f}, p-value = {p_value:.4f}")
        return p_value > 0.05

    def perform_anova(self, continuous_var, categorical_var):
        """Perform ANOVA test."""
        groups = [self.df[continuous_var][self.df[categorical_var] == cat].dropna() for cat in self.df[categorical_var].unique()]
        if self.check_normality(continuous_var):
            f_stat, p_value = stats.f_oneway(*groups)
            print(f"ANOVA Test: F-statistic = {f_stat:.4f}, p-value = {p_value:.4f}")
        else:
            h_stat, p_value = stats.kruskal(*groups)
            print(f"Kruskal-Wallis Test: H-statistic = {h_stat:.4f}, p-value = {p_value:.4f}")

    def perform_t_test(self, continuous_var, categorical_var):
        """Perform t-test."""
        groups = [self.df[continuous_var][self.df[categorical_var] == cat].dropna() for cat in self.df[categorical_var].unique()]
        if len(groups) == 2:  # Ensure only two groups for t-test
            if self.check_normality(continuous_var):
                stat, p_value = stats.ttest_ind(*groups, equal_var=False)
                print(f"T-test: Statistic = {stat:.4f}, p-value = {p_value:.4f}")
            else:
                stat, p_value = stats.mannwhitneyu(*groups)
                print(f"Mann-Whitney U Test: Statistic = {stat:.4f}, p-value = {p_value:.4f}")
        else:
            print("T-test requires exactly two groups.")

    def perform_chi_square(self, var1, var2):
        """Perform Chi-Square test."""
        contingency_table = pd.crosstab(self.df[var1], self.df[var2])
        chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"Chi-Square Test: Statistic = {chi2_stat:.4f}, p-value = {p_value:.4f}")

    def perform_regression(self, dependent_var, independent_var):
        """Perform regression analysis."""
        model = ols(f"{dependent_var} ~ {independent_var}", data=self.df).fit()
        print(model.summary())

# Sentiment Analysis class from gex5.py
class SentimentAnalysis:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)

    def get_text_columns(self):
        """Identifies and returns text columns."""
        text_columns = self.df.select_dtypes(include=['object'])
        return text_columns.columns.tolist()

    def vader_sentiment_analysis(self, data):
        analyzer = SentimentIntensityAnalyzer()
        scores = []
        sentiments = []
        for text in data:
            score = analyzer.polarity_scores(text)['compound']
            scores.append(score)
            sentiments.append('positive' if score > 0.05 else ('negative' if score < -0.05 else 'neutral'))
        return scores, sentiments

    def textblob_sentiment_analysis(self, data):
        scores = []
        sentiments = []
        for text in data:
            analysis = TextBlob(text)
            scores.append(analysis.sentiment.polarity)
            sentiments.append('positive' if scores[-1] > 0 else ('neutral' if scores[-1] == 0 else 'negative'))
        return scores, sentiments

# Main function to interact with the user
def main():
    file_path = input("Enter the path to your dataset (e.g., ParisHousing_rearranged.csv): ")
    analysis = DataAnalysis(file_path)
    
    while True:
        print("\nHow do you want to analyze your data?")
        print("1. Plot variable distribution")
        print("2. Conduct ANOVA")
        print("3. Conduct t-Test")
        print("4. Conduct Chi-Square")
        print("5. Conduct Regression")
        print("6. Conduct Sentiment Analysis")
        print("7. Quit")
        
        choice = input("Enter your choice (1 – 7): ")
        
        if choice == '1':
            variable = input("Enter the variable you want to plot (e.g., Age, Income): ")
            analysis.plot_distribution(variable)

        elif choice == '2':
            continuous_var = input("Enter a continuous (interval/ratio) variable (e.g., Age, Income): ")
            categorical_var = input("Enter a categorical (ordinal/nominal) variable (e.g., Income Group): ")
            analysis.perform_anova(continuous_var, categorical_var)

        elif choice == '3':
            continuous_var = input("Enter a continuous variable for t-Test: ")
            categorical_var = input("Enter a categorical variable: ")
            analysis.perform_t_test(continuous_var, categorical_var)

        elif choice == '4':
            var1 = input("Enter the first categorical variable: ")
            var2 = input("Enter the second categorical variable: ")
            analysis.perform_chi_square(var1, var2)

        elif choice == '5':
            dependent_var = input("Enter the dependent variable: ")
            independent_var = input("Enter the independent variable: ")
            analysis.perform_regression(dependent_var, independent_var)

        elif choice == '6':
            sentiment_analysis = SentimentAnalysis(file_path)
            text_columns = sentiment_analysis.get_text_columns()
            print("Available text columns for sentiment analysis:")
            for i, col in enumerate(text_columns):
                print(f"{i + 1}: {col}")
            column_index = int(input("Select the column number for sentiment analysis: ")) - 1
            column = text_columns[column_index]

            print("Choose the type of sentiment analysis to perform:")
            print("1. VADER")
            print("2. TextBlob")
            sentiment_choice = input("Enter the number corresponding to your choice: ")

            if sentiment_choice == '1':
                scores, sentiments = sentiment_analysis.vader_sentiment_analysis(sentiment_analysis.df[column])
                result_df = pd.DataFrame({column: sentiment_analysis.df[column], 'VADER Score': scores, 'Sentiment': sentiments})
            elif sentiment_choice == '2':
                scores, sentiments = sentiment_analysis.textblob_sentiment_analysis(sentiment_analysis.df[column])
                result_df = pd.DataFrame({column: sentiment_analysis.df[column], 'TextBlob Polarity': scores, 'Sentiment': sentiments})

            print(result_df)

        elif choice == '7':
            print("Exiting the program.")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
