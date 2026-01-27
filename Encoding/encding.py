from sklearn.preprocessing import  LabelEncoder
import pandas as pd

df = pd.DataFrame({
    'Color': ['Red', 'Blue', 'Green', 'Blue', 'Red'], 
    'Size': ['S', 'M', 'L', 'XL', 'S']
})  

df_label_encoded = df.copy()

le = LabelEncoder()
df_label_encoded['Color'] = le.fit_transform(df['Color'])
df_label_encoded['Size'] = le.fit_transform(df['Size'])     

print(df_label_encoded)
print(df_label_encoded[['Color', 'Size']].dtypes)

