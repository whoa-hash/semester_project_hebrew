import unicodedata


def remover_nkd(text):
    nikud = text
    normalized = unicodedata.normalize('NFKD', nikud)  # Reduce hebrew vowel ניקוד marks
    removed_nikud = ""
    # since hebrew reads the other way compared to english
    for char in range(len(normalized), 0, -1):
        character = normalized[char - 1]
        if not unicodedata.combining(character):
            removed_nikud = character + removed_nikud
    return removed_nikud
