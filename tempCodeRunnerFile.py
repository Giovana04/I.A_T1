    for col in df.columns: # Preenche valores ausentes com a mediana (numérico) ou moda (categórico) - no caso o nosso ta tudo preenchido mas é bom ter
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0],inplace=True)
        else:
            df[col].fillna(df[col].median(),inplace=True)