import pandas as pd

df1 = pd.read_csv('prvi_fajl.csv')
df2 = pd.read_csv('drugi_fajl.csv')

# Unesi ime kolone po kojoj proveravaš preklapanje (npr. 'id')
kolona_za_proveru = 'id' 

# Filtriramo df2: uzimamo samo one redove čiji 'id' NE POSTOJI u df1
filtrirani_df2 = df2[~df2[kolona_za_proveru].isin(df1[kolona_za_proveru])]

# Sada spajamo ceo prvi fajl i samo "čiste" nove podatke iz drugog fajla
konacni_df = pd.concat([df1, filtrirani_df2], ignore_index=True)

# Čuvanje rezultata
konacni_df.to_csv('spojeni_bez_preklapanja.csv', index=False)
print("Uspešno! Iz drugog fajla su dodati samo jedinstveni podaci koji ne postoje u prvom.")