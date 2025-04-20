import pandas as pd

class OptimalThroughputProcessor:
    def __init__(self, input_csv: str, output_csv: str, quantile: float = 0.65):
        """
        Initializes the processor with input/output paths and the quantile threshold.
        
        :param input_csv: Path to the input CSV file.
        :param output_csv: Path where the processed CSV will be saved.
        :param quantile: Quantile to use for thresholding (default: 0.65).
        """
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.quantile = quantile
        self.df = None

    def load_data(self):
        """Loads the CSV into a DataFrame."""
        self.df = pd.read_csv(self.input_csv)
        print(f"Loaded data from {self.input_csv}, shape = {self.df.shape}")

    def label_access_count(self):
        """Adds 'access_count_label' based on 'access_count' ranges."""
        self.df['access_count_label'] = self.df['access_count'].apply(
            lambda x: 1 if x <= 10 else (2 if x <= 20 else 3)
        )
        print("Added 'access_count_label' column.")

    def build_combination(self):
        """Creates the 'combination' column by concatenating file size and access count label."""
        self.df['combination'] = (
            self.df['file_size_KB'].astype(str)
            + ' | '
            + self.df['access_count_label'].astype(str)
        )
        print("Built 'combination' column.")

    def compute_OT(self):
        """
        Computes the threshold per combination and
        adds the 'OT' column: 1 if throughput >= threshold, else 0.
        """
        # Compute per-group quantile threshold
        self.df['threshold'] = self.df.groupby('combination')['throughput_KBps']\
                                      .transform(lambda x: x.quantile(self.quantile))
        # Fill OT based on comparison
        self.df['OT'] = (self.df['throughput_KBps'] >= self.df['threshold']).astype(int)
        # Clean up
        self.df.drop(columns='threshold', inplace=True)
        print(f"Computed 'OT' column using the {self.quantile * 100}th percentile threshold.")

    def save_data(self):
        """Saves the processed DataFrame to the output CSV."""
        self.df.to_csv(self.output_csv, index=False)
        print(f"Saved processed data to {self.output_csv}")

    def run(self):
        """Executes the full processing pipeline."""
        self.load_data()
        self.label_access_count()
        self.build_combination()
        self.compute_OT()
        self.save_data()

