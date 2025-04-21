from beechunker.ml.random_forest import BeeChunkerRF
import pandas as pd
from beechunker.optimizer.chunk_manager import ChunkSizeOptimizer

df = pd.read_csv("/home/jgajbha/BeeChunker/test.csv")

def main():
    # Initialize the RandomForest class
    rf = BeeChunkerRF()
    
    # Predict the chunk size using the trained model
    
    predictions = rf.predict(df)
    print(predictions)
    optimal_chunk_size = int(predictions.iloc[0]['optimal_chunk_KB'])
    print(f"Optimal chunk size: {optimal_chunk_size} KB")

def test_optimizer():
    optimizer = ChunkSizeOptimizer()
    file_path = "/mnt/beegfs/dummy_file.txt"
    
    # First, get the current chunk size to make sure we have it
    current_chunk_size = optimizer.get_current_chunk_size(file_path)
    if current_chunk_size is None:
        print(f"Could not determine current chunk size for {file_path}")
        return
    
    # Now run the optimization
    result = optimizer.optimize_file(file_path, model_type="rf")
    print(result)
    
    
if __name__ == "__main__":
    # test_optimizer()
    main()