def type_of_feature(df, threshold = 1000):
    num_cols = []
    cat_cols = []
    for col in df.columns:
        if df[col].nunique() < threshold:
            cat_cols.append(col)
        else:
            num_cols.append(col)
    return num_cols, cat_cols