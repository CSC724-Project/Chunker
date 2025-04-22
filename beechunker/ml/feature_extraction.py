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
        # print(f"Loaded data from {self.input_csv}, shape = {self.df.shape}")

    def label_access_count(self):
        """Adds 'access_count_label' based on 'access_count' ranges."""
        self.df['access_count_label'] = self.df['access_count'].apply(
            lambda x: 1 if x <= 10 else (2 if x <= 20 else 3)
        )
        # print("Added 'access_count_label' column.")

    def build_combination(self):
        """Creates the 'combination' column by concatenating file size and access count label."""
        self.df['combination'] = (
            self.df['file_size_KB'].astype(str)
            + ' | '
            + self.df['access_count_label'].astype(str)
        )
        # print("Built 'combination' column.")

    def compute_OT(self):
        """
        Computes the threshold per combination and
        adds the 'OT' column: 1 if throughput >= threshold, else 0.
        """
        # Make sure throughput_KBps has no missing values
        if self.df['throughput_KBps'].isna().any():
            print(f"Warning: Found {self.df['throughput_KBps'].isna().sum()} missing throughput values")
            self.df = self.df.dropna(subset=['throughput_KBps'])
            print(f"Removed rows with missing throughput values, new shape = {self.df.shape}")

        # 1. Create the 'threshold' column first using groupby and transform
        # Use a lambda function to handle potential empty/small groups gracefully
        self.df['threshold'] = self.df.groupby('combination')['throughput_KBps'].transform(
            lambda x: x.quantile(0.65) if len(x) >= 1 else None # Calculate quantile if group has at least 1 member
        )
        # print(f"Initial threshold calculation done. Missing thresholds: {self.df['threshold'].isna().sum()}")

        # 2. Now fill any missing thresholds (NaN values resulting from small/empty groups)
        if self.df['threshold'].isna().any():
            missing_count = self.df['threshold'].isna().sum()
            print(f"Warning: {missing_count} combinations resulted in NaN threshold (likely small groups). Filling with global quantile.")
            global_quantile = self.df['throughput_KBps'].quantile(self.quantile)
            self.df['threshold'].fillna(global_quantile, inplace=True)
            print(f"Filled {missing_count} missing thresholds with global quantile: {global_quantile}")

        # Ensure no NaNs remain before comparison
        if self.df['threshold'].isna().any():
            # This should ideally not happen after fillna, but as a safeguard
            print("Error: Still found NaN values in threshold after filling. Check data or logic.")
            # Handle this case, e.g., fill with 0 or raise an error
            self.df['threshold'].fillna(0, inplace=True) # Example: fill with 0

        # 3. Fill OT based on comparison
        # Ensure both columns are numeric before comparison
        self.df['throughput_KBps'] = pd.to_numeric(self.df['throughput_KBps'], errors='coerce')
        self.df['threshold'] = pd.to_numeric(self.df['threshold'], errors='coerce')
        # Drop rows where conversion might have failed (optional, depends on desired behavior)
        self.df.dropna(subset=['throughput_KBps', 'threshold'], inplace=True)

        # print(f"Comparing throughput (type: {self.df['throughput_KBps'].dtype}) with threshold (type: {self.df['threshold'].dtype})")
        self.df['OT'] = (self.df['throughput_KBps'] >= self.df['threshold']).astype(int)

        # 4. Clean up
        self.df.drop(columns='threshold', inplace=True)
        # print(f"Computed 'OT' column using the {0.65 * 100}th percentile threshold.")
    
    def save_data(self):
        """Saves the processed DataFrame to the output CSV."""
        self.df.to_csv(self.output_csv, index=False)
        # print(f"Saved processed data to {self.output_csv}")

    def run(self):
        """Executes the full processing pipeline."""
        self.load_data()
        self.label_access_count()
        self.build_combination()
        self.compute_OT()
        self.save_data()

