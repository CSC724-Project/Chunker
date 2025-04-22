from beechunker.optimizer.chunk_manager import ChunkSizeOptimizer

def main():
    """Check the chunk size optimization."""
    optimizer = ChunkSizeOptimizer()
    result = optimizer.optimize_file("/mnt/beegfs/dummy_file.txt", model_type="xgb")
    print(result)
    
if __name__ == "__main__":
    main()