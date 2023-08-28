# -*- coding: utf-8 -*-
from __future__ import annotations

class Bin:
    def __init__(self, size):
        """Initialise new bin."""
        self._size = size
        self._size_used = 0
        self._items = []
    
    def can_fit(self, item_size):
        """Returns True if the item can fit in the bin, False otherwise."""
        return item_size <= (self._size - self._size_used)
    
    def put(self, item, item_size):
        """Put the item with the given item size in the bin."""
        if not self.can_fit(item_size):
            raise ValueError("Item does not fit in bin.")
        self._items.append(item_size)
        self._size_used += item_size

def ffd(items: list[tuple], max_bins: int, bin_size: int) -> list['Bin'] | None:
    """ 
    Perform the first-fit decreasing heuristic algorithm. Try to pack item into
    bins. If packable returns the packed bins.
    
    Parameters:
        items - List of items where items are represented as tuples. The tuples 
            are in the form (item_id, item_size).
        max_bins - The maximum number of bins we are allowed to open.
        bin_size - The size of each bin.
    """
    # Sort items by decreasing size
    items.sort(key=lambda x: x[1], reverse=True)
    open_bins = []
    
    # Try and fit items into bins
    for item in items:
        item_id, item_size = item
        
        # Try to pack into an existing bin
        for bin_ in open_bins:
            if bin_.can_fit(item_size):
                bin_.put(item, item_size)
                continue
        
        # No bins left to open
        if len(open_bins) == max_bins:
            return None
        
        # Open new bin
        new_bin = Bin(bin_size)
        new_bin.put(item, item_size)
        open_bins.append(new_bin)
    
    return open_bins
    
        