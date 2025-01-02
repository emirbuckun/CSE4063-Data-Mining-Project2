from collections import defaultdict
from typing import List, Set, Dict, Tuple
import pandas as pd
import numpy as np
from time import time

class ECLAT:
    def __init__(self, min_support: float = 0.1):
        """
        Initialize ECLAT algorithm
        
        Parameters:
        -----------
        min_support : float
            Minimum support threshold (between 0 and 1)
        """
        self.min_support = min_support
        self.transaction_count = 0
        self.min_support_count = 0
        self.frequent_itemsets = []
        self.execution_time = 0
        
    def fit(self, df: pd.DataFrame) -> List[Tuple[frozenset, int]]:
        """
        Find frequent itemsets using ECLAT algorithm
        
        Parameters:
        -----------
        df : pandas DataFrame
            Binary transaction data where columns are items and rows are transactions
            
        Returns:
        --------
        List[Tuple[frozenset, int]]
            List of tuples containing frequent itemsets and their support counts
        """
        start_time = time()
        
        # Initialize parameters
        self.transaction_count = len(df)
        self.min_support_count = int(self.min_support * self.transaction_count)
        
        # Convert DataFrame to vertical format (item: set of transaction IDs)
        vertical_db = self._create_vertical_database(df)
        
        # Find frequent 1-itemsets and their tidsets
        frequent_items = {
            frozenset([item]): tidset 
            for item, tidset in vertical_db.items() 
            if len(tidset) >= self.min_support_count
        }
        
        # Store results
        self.frequent_itemsets = [(itemset, len(tidset)) 
                                for itemset, tidset in frequent_items.items()]
        
        # Find frequent itemsets of size > 1
        self._eclat(frequent_items)
        
        self.execution_time = time() - start_time
        
        # Sort by support count and itemset size
        self.frequent_itemsets.sort(key=lambda x: (-x[1], len(x[0])))
        return self.frequent_itemsets
    
    def _create_vertical_database(self, df: pd.DataFrame) -> Dict[str, Set[int]]:
        """Convert horizontal database to vertical format"""
        vertical_db = defaultdict(set)
        for tid, row in df.iterrows():
            for item, val in row.items():
                if val == 1:  # Only consider present items (1s)
                    vertical_db[item].add(tid)
        return vertical_db
    
    def _eclat(self, frequent_items: Dict[frozenset, Set[int]], k: int = 2):
        """Recursive ECLAT implementation"""
        if not frequent_items:
            return
        
        # Generate candidate (k+1)-itemsets
        next_frequent_items = {}
        items = list(frequent_items.keys())
        
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                # Get k-1 common items
                itemset1 = items[i]
                itemset2 = items[j]
                if len(itemset1.union(itemset2)) == k:
                    # Compute intersection of tidsets
                    new_tidset = frequent_items[itemset1].intersection(frequent_items[itemset2])
                    
                    # Check support
                    if len(new_tidset) >= self.min_support_count:
                        new_itemset = itemset1.union(itemset2)
                        next_frequent_items[new_itemset] = new_tidset
                        self.frequent_itemsets.append((new_itemset, len(new_tidset)))
        
        # Recursive call for next level
        self._eclat(next_frequent_items, k + 1)
    
    def get_support(self, itemset: frozenset) -> float:
        """Get support value for an itemset"""
        for items, count in self.frequent_itemsets:
            if items == itemset:
                return count / self.transaction_count
        return 0.0
    
    def get_frequent_itemsets_df(self) -> pd.DataFrame:
        """Return frequent itemsets as a DataFrame"""
        itemsets = []
        supports = []
        
        for itemset, count in self.frequent_itemsets:
            itemsets.append(list(itemset))
            supports.append(count / self.transaction_count)
            
        return pd.DataFrame({
            'itemsets': itemsets,
            'support': supports
        }).sort_values('support', ascending=False)

def prepare_olympic_data_for_eclat(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare Olympic data for ECLAT algorithm
    
    Parameters:
    -----------
    df : pandas DataFrame
        Preprocessed Olympic data
        
    Returns:
    --------
    pandas DataFrame
        Binary transaction data suitable for ECLAT
    """
    # Create transactions based on Games and Athletes
    transactions = df.groupby(['Games', 'Team'])[['Sport', 'Medal']].agg(
        lambda x: list(x.dropna())
    ).reset_index()
    
    # Create binary columns for each unique Sport-Medal combination
    all_items = set()
    for items in transactions.iloc[:, 2:].values.flatten():
        all_items.update(items)
    
    # Create binary DataFrame
    binary_df = pd.DataFrame(index=transactions.index)
    for item in all_items:
        binary_df[str(item)] = transactions.apply(
            lambda row: 1 if item in row['Sport'] or item in row['Medal'] else 0,
            axis=1
        )
    
    return binary_df

# Example usage:
if __name__ == "__main__":
    # Create sample data
    data = {
        'item1': [1, 0, 1, 1, 0],
        'item2': [1, 1, 0, 1, 1],
        'item3': [1, 1, 1, 0, 0],
        'item4': [0, 1, 0, 1, 1]
    }
    df = pd.DataFrame(data)
    
    # Initialize and run ECLAT
    eclat = ECLAT(min_support=0.4)
    frequent_itemsets = eclat.fit(df)
    
    # Print results
    print("\nFrequent Itemsets:")
    for itemset, support_count in frequent_itemsets:
        print(f"Items: {list(itemset)}, Support Count: {support_count}")
    
    print(f"\nExecution time: {eclat.execution_time:.4f} seconds")