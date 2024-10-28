class AttributeDict:
    def __init__(self, dictionary=None):
        """
        Initialize with an optional dictionary.
        If no dictionary is provided, starts empty.
        """
        # Initialize the _data attribute first
        super().__setattr__('_data', {})
        
        # Then update with the provided dictionary
        if dictionary is not None:
            for key, value in dictionary.items():
                setattr(self, key, value)
    
    def __getattr__(self, name):
        """
        Allows accessing dictionary items as attributes.
        Raises AttributeError if key doesn't exist.
        """
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"'AttributeDict' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        """
        Allows setting dictionary items as attributes.
        Special case for '_data' which is the internal dictionary.
        """
        if name == '_data':
            super().__setattr__(name, value)
        else:
            self._data[name] = value
    
    def __getitem__(self, key):
        """Allows dictionary-style access"""
        return self._data[key]
    
    def __setitem__(self, key, value):
        """Allows dictionary-style setting"""
        self._data[key] = value
    
    def update(self, dictionary):
        """Update with another dictionary"""
        for key, value in dictionary.items():
            setattr(self, key, value)
    
    def to_dict(self):
        """Convert back to a regular dictionary"""
        return dict(self._data)
    
    def __dict__(self):
        """Support for built-in dict() conversion"""
        return self._data
    
    def __iter__(self):
        """Make the object iterable like a dictionary"""
        return iter(self._data)
    
    def items(self):
        """Support for .items() method like a regular dictionary"""
        return self._data.items()
    
    def __repr__(self):
        """String representation"""
        return f"AttributeDict({self._data})"