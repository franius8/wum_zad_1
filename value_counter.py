import pandas as pd

def count_values_in_column(file_path, separator=';'):
    """
    Reads a CSV file and counts occurrences of values in a user-selected column.
    
    Args:
        file_path (str): Path to the CSV file
        separator (str): Delimiter used in the CSV file
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path, sep=separator)
        
        # Display available columns
        print("\nAvailable columns:")
        for i, column in enumerate(df.columns):
            print(f"{i+1}. {column}")
        
        # Get user input for column selection
        while True:
            try:
                choice = int(input("\nEnter the number of the column you want to analyze: "))
                if 1 <= choice <= len(df.columns):
                    selected_column = df.columns[choice-1]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(df.columns)}")
            except ValueError:
                print("Please enter a valid number")
        
        # Count values in the selected column
        value_counts = df[selected_column].value_counts().sort_values(ascending=False)
        
        # Display results
        print(f"\nValue counts for column '{selected_column}':")
        print(value_counts)
        
        # Display some statistics
        print(f"\nTotal unique values: {len(value_counts)}")
        print(f"Most common value: {value_counts.index[0]} (appears {value_counts.iloc[0]} times)")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    file_path = 'earnings.csv'
    count_values_in_column(file_path)