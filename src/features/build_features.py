def reverse_one_hot_encoding(encoding):
    """
    Funktion, die das One-Hot-Encoding zurücksetzt und eine Liste von Kategorien zurückgibt.
    """
    # ein leeres Set für die einzigartigen Kategorien erstellen
    unique_categories = set()
    
    # die einzigartigen Kategorien aus der binären Matrix extrahieren
    for binary in encoding:
        unique_categories.add(tuple(binary))
    
    # eine Liste der Kategorien erstellen, indem jedes Element der binären Matrix mit der entsprechenden Kategorie abgeglichen wird
    categories = []
    for binary in encoding:
        for category in unique_categories:
            if tuple(binary) == category:
                categories.append(list(unique_categories).index(category))
    
    # die Liste der Kategorien zurückgeben
    return categories
    
def is_one_hot_encoded(df):
	"""
    Check whether the submitted dataframe is one-hot encoded or not. Each row
    of a one-hot encoded dataframe has to contain exactly one occurence of 1,
    while the remaining entries are 0.

    :param df: the dataframe to be examined
    :return: whether the dataframe is one-hot encoded or not
    """ 
	
	# check whether df only contains 0 and 1
	if not df.isin([0, 1]).all(None):
		return False
		
	# check whether 1 occurs exactly once per row
	row_sums = df.sum(axis=1)
	return row_sums.isin([1]).all()
	
def revert_one_hot_encoding(df):
	"""
    Reverts a one-hot encoded dataframe by returning the feature with the
    highest value for each row. In a one-hot encoded dataframe this is the
    feature with the value 1, while the remaining features have the value 0.

    :param df: the dataframe to be decoded
    :return: the decoded dataframe
    """ 
	return df.idxmax(axis=1)
