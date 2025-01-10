import time
import pandas as pd
from collections import defaultdict

class ECLAT:
    def __init__(self, min_support=0.1):
        """
        Initialize ECLAT algorithm
        
        Parameters:
        -----------
        min_support : float
            Minimum support threshold (between 0 and 1)
        """
        self.min_support = min_support
        self.transaction_count = 0
        self.itemsets = []
        self.item_tids = {}
        self.execution_time = 0

    def _create_vertical_database(self, df):
        """
        Convert horizontal database to vertical format (item: set of transaction IDs)
        """
        tid_dict = defaultdict(set)
        for idx, row in df.iterrows():
            for item, val in row.items():
                if val:  # Only consider True values for boolean data
                    tid_dict[item].add(idx)
        return tid_dict

    def _merge_itemsets(self, itemset1, itemset2, tids1, tids2):
        """
        Merge two itemsets and their transaction IDs if they share k-1 items
        """
        if len(itemset1) == 1 or self._share_prefix(itemset1, itemset2):
            new_itemset = itemset1 | itemset2
            new_tids = tids1 & tids2
            support = len(new_tids) / self.transaction_count
            if support >= self.min_support:
                return new_itemset, new_tids
        return None, None

    def _share_prefix(self, itemset1, itemset2):
        """
        Check if two itemsets share the same prefix except last item
        """
        if len(itemset1) == 1:
            return True
        sorted1 = sorted(list(itemset1)[:-1])
        sorted2 = sorted(list(itemset2)[:-1])
        return sorted1 == sorted2

    def _eclat_recursive(self, prefix_itemsets):
        """
        Recursive ECLAT implementation
        """
        items = list(prefix_itemsets.items())
        for i in range(len(items)):
            itemset1, tids1 = items[i]
            if not isinstance(itemset1, frozenset):
                itemset1 = frozenset([itemset1])
            
            for j in range(i + 1, len(items)):
                itemset2, tids2 = items[j]
                if not isinstance(itemset2, frozenset):
                    itemset2 = frozenset([itemset2])
                
                new_itemset, new_tids = self._merge_itemsets(itemset1, itemset2, tids1, tids2)
                if new_itemset is not None:
                    self.itemsets.append((new_itemset, len(new_tids)))
                    new_prefix_itemsets = {new_itemset: new_tids}
                    self._eclat_recursive(new_prefix_itemsets)

    def fit(self, df):
        """
        Find frequent itemsets using ECLAT algorithm
        
        Parameters:
        -----------
        df : pandas DataFrame
            Boolean transaction data where columns are items
            
        Returns:
        --------
        list
            List of tuples (itemset, support_count)
        """
        start_time = time.time()
        
        self.transaction_count = len(df)
        self.itemsets = []
        
        # Create vertical database
        item_tids = self._create_vertical_database(df)
        
        # Filter items by minimum support and convert to frozenset
        prefix_itemsets = {}
        for item, tids in item_tids.items():
            support = len(tids) / self.transaction_count
            if support >= self.min_support:
                self.itemsets.append((frozenset([item]), len(tids)))
                prefix_itemsets[item] = tids
        
        # Find frequent itemsets recursively
        self._eclat_recursive(prefix_itemsets)
        
        # Sort by support count and itemset size
        self.itemsets.sort(key=lambda x: (-x[1], len(x[0])))
        
        self.execution_time = time.time() - start_time
        
        return self.itemsets

    def get_support(self, itemset):
        """
        Get support value for an itemset
        """
        if isinstance(itemset, (list, set)):
            itemset = frozenset(itemset)
        for items, count in self.itemsets:
            if items == itemset:
                return count / self.transaction_count
        return 0.0

    def get_frequent_itemsets_df(self):
        """
        Return frequent itemsets as a DataFrame
        """
        itemsets = []
        supports = []
        sizes = []
        
        for itemset, count in self.itemsets:
            itemsets.append(list(itemset))
            supports.append(count / self.transaction_count)
            sizes.append(len(itemset))
            
        return pd.DataFrame({
            'itemsets': itemsets,
            'support': supports,
            'size': sizes
        }).sort_values(['support', 'size'], ascending=[False, True])

def prepare_olympic_data_for_eclat(df):
    """
    Prepare Olympic data for ECLAT algorithm
    """
    # Create binary columns for Sport and Medal combinations
    sport_dummies = pd.get_dummies(df['Sport'], prefix='Sport')
    medal_dummies = pd.get_dummies(df['Medal'].fillna('No_Medal'), prefix='Medal')
    
    # Combine all binary columns
    eclat_data = pd.concat([sport_dummies, medal_dummies], axis=1).astype(bool)
    
    return eclat_data