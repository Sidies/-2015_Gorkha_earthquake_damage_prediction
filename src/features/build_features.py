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
