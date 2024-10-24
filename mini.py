import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
try:
    from transformers import pipeline
except ImportError:
    pipeline = None

class DataAnalysis:
    def __init__(self):
        self.df = None
        self.column_types = None

    def load_data(self, path):
        """Loads the dataset from the given path."""
        self.df = pd.read_csv(path)
        print(f"Dataset loaded successfully. {self.df.shape[0]} rows and {self.df.shape[1]} columns.")

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
                elif unique_values == 2:
                    column_types['nominal'].append(col)
                else:
                    column_types['numeric_ordinal'].append(col)
            elif pd.api.types.is_bool_dtype(self.df[col]) or pd.api.types.is_object_dtype(self.df[col]):
                column_types['nominal'].append(col)
            else:
                column_types['non_numeric_ordinal'].append(col)

        self.column_types = column_types
        return column_types

    def select_variable(self, data_type):
        """Selects a variable based on data type."""
        available_columns = self.column_types[data_type]
        print(f"\nAvailable {data_type} variables:")
        for i, col in enumerate(available_columns):
            print(f"{i + 1}. {col} ({self.df[col].nunique()} unique values)")

        choice = int(input(f"Select the {data_type} variable index: ")) - 1
        return available_columns[choice]

    def check_normality(self, column):
        """Checks normality using Shapiro-Wilk test and generates a Q-Q plot."""
        data = self.df[column].dropna()
        sm.qqplot(data, line='s')
        plt.title(f"Q-Q Plot for {column}")
        plt.show()

        stat, p_value = stats.shapiro(data)
        print(f"Shapiro-Wilk Test: Statistic = {stat:.4f}, p-value = {p_value:.4f}")
        return p_value

    def perform_anova(self, continuous_var, categorical_var):
        """Performs ANOVA test and checks normality."""
        groups = [group[continuous_var].dropna() for name, group in self.df.groupby(categorical_var)]
        normality_p = self.check_normality(continuous_var)

        if normality_p > 0.05:
            f_stat, p_value = stats.f_oneway(*groups)
            print(f"ANOVA: F-statistic = {f_stat:.4f}, p-value = {p_value:.4f}")
        else:
            print(f"'{continuous_var}' is not normally distributed, using Kruskal-Wallis Test instead.")
            h_stat, p_value = stats.kruskal(*groups)
            print(f"Kruskal-Wallis Test: H-statistic = {h_stat:.4f}, p-value = {p_value:.4f}")

    def perform_regression(self, x_col, y_col):
        """Performs linear regression and checks for normality."""
        x = self.df[x_col].dropna()
        y = self.df[y_col].dropna()
        valid_data = pd.concat([x, y], axis=1).dropna()

        if valid_data.empty:
            print("Error: No valid data points for regression.")
            return

        self.check_normality(y_col)
        slope, intercept, r_value, p_value, std_err = stats.linregress(valid_data[x_col], valid_data[y_col])
        print(f"Slope: {slope}, Intercept: {intercept}, R-squared: {r_value**2}, p-value: {p_value}")

    def t_test_or_mannwhitney(self, continuous_var, categorical_var):
        """Perform t-test or Mann-Whitney U test based on normality."""
        groups = self.df.groupby(categorical_var)[continuous_var].apply(list)
        normality_p = self.check_normality(continuous_var)

        if normality_p > 0.05:
            stat, p_value = stats.ttest_ind(*groups, equal_var=False)
            test_name = "t-test"
        else:
            stat, p_value = stats.mannwhitneyu(*groups)
            test_name = "Mann-Whitney U test"

        print(f"{test_name}: Statistic = {stat:.4f}, p-value = {p_value:.4f}")

    def chi_square_test(self, var1, var2):
        """Performs Chi-square test."""
        contingency_table = pd.crosstab(self.df[var1], self.df[var2])
        stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        print(f"Chi-square Test: Statistic = {stat:.4f}, p-value = {p_value:.4f}")

        contingency_table.plot(kind='bar', stacked=True)
        plt.title('Stacked Bar Chart')
        plt.xlabel(var1)
        plt.ylabel('Frequency')
        plt.show()

    def plot_distribution(self, variable):
        """Plots histogram for the given variable."""
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df[variable].dropna(), bins=30, kde=True)
        plt.title(f'Distribution of {variable}')
        plt.xlabel(variable)
        plt.ylabel('Frequency')
        plt.show()

class SentimentAnalysis:
    def __init__(self):
        self.df = None

    def load_data(self, path):
        """Loads the dataset from the given path."""
        self.df = pd.read_csv(path)

    def vader_sentiment_analysis(self, data):
        """Perform VADER sentiment analysis on the provided text data."""
        analyzer = SentimentIntensityAnalyzer()
        scores = []
        sentiments = []
        for text in data:
            score = analyzer.polarity_scores(text)['compound']
            scores.append(score)
            if score >= 0.05:
                sentiments.append('positive')
            elif score <= -0.05:
                sentiments.append('negative')
            else:
                sentiments.append('neutral')
        return scores, sentiments

    def textblob_sentiment_analysis(self, data):
        """Perform TextBlob sentiment analysis on the provided text data."""
        scores = []
        sentiments = []
        for text in data:
            analysis = TextBlob(text)
            score = analysis.sentiment.polarity
            scores.append(score)
            sentiments.append('positive' if score > 0 else 'negative' if score < 0 else 'neutral')
        return scores, sentiments

    def distilbert_sentiment_analysis(self, data):
        """Perform DistilBERT sentiment analysis on the provided text data."""
        if not pipeline:
            raise ImportError("transformers library is not installed.")
        
        bert_pipeline = pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')
        scores = []
        sentiments = []
        for text in data:
            result = bert_pipeline(text)[0]
            label = result['label']
            scores.append(result['score'])
            sentiments.append('positive' if label in ['4 stars', '5 stars'] else 'negative' if label in ['1 star', '2 stars'] else 'neutral')
        return scores, sentiments

def main():
    # Load dataset
    path = input("Enter the path to the CSV file: ")
    analysis = DataAnalysis()
    analysis.load_data(path)
    analysis.list_column_types()

    # Choose analysis
    while True:
        print("\nHow do you want to analyze your data?")
        print("1. Plot variable distribution")
        print("2. Conduct ANOVA")
        print("3. Conduct t-Test")
        print("4. Conduct chi-Square")
        print("5. Conduct Regression")
        print("6. Conduct Sentiment Analysis")
        print("7. Quit")
        choice = input("Enter your choice (1 – 7): ")

        if choice == '1':
            variable = analysis.select_variable('interval')
            analysis.plot_distribution(variable)

        elif choice == '2':
            continuous_var = analysis.select_variable('interval')
            categorical_var = analysis.select_variable('nominal')
            analysis.perform_anova(continuous_var, categorical_var)

        elif choice == '3':
            continuous_var = analysis.select_variable('interval')
            categorical_var = analysis.select_variable('nominal')
            analysis.t_test_or_mannwhitney(continuous_var, categorical_var)

        elif choice == '4':
            var1 = analysis.select_variable('nominal')
            var2 = analysis.select_variable('nominal')
            analysis.chi_square_test(var1, var2)

        elif choice == '5':
            x_var = analysis.select_variable('interval')
            y_var = analysis.select_variable('interval')
            analysis.perform_regression(x_var, y_var)

        elif choice == '6':
            sa = SentimentAnalysis()
            sa.load_data(path)
            text_columns = sa.df.select_dtypes(include=['object']).columns.tolist()
            print("\nText columns available for sentiment analysis:", text_columns)
            text_var = input("Select a text column for sentiment analysis: ")
            print("\n1. VADER Sentiment Analysis")
            print("2. TextBlob Sentiment Analysis")
            print("3. DistilBERT Sentiment Analysis")
            sa_choice = input("Choose sentiment analysis method (1-3): ")

            if sa_choice == '1':
                scores, sentiments = sa.vader_sentiment_analysis(sa.df[text_var])
            elif sa_choice == '2':
                scores, sentiments = sa.textblob_sentiment_analysis(sa.df[text_var])
            elif sa_choice == '3':
                scores, sentiments = sa.distilbert_sentiment_analysis(sa.df[text_var])
            else:
                print("Invalid choice.")
                continue

            sa.df[f'{text_var}_sentiment_score'] = scores
            sa.df[f'{text_var}_sentiment'] = sentiments
            print(sa.df[[text_var, f'{text_var}_sentiment_score', f'{text_var}_sentiment']])

        elif choice == '7':
            print("Exiting...")
            break

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
