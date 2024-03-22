import pandas as pd
from sklearn.utils import resample

def balance_classes(df, target_column,val):
    seed = 42
    # Count the number of observations in each class
    class_counts = df[target_column].value_counts()
    
    # Desired value of observations in each class
    desired_value = val
    
    # Create a list to store the DataFrame slices
    df_list = []
    
    # Loop through each class and sample
    for class_label in class_counts.index:
        df_class = df[df[target_column] == class_label]
        if class_counts[class_label] > desired_value:
            # If the class is larger than desired, undersample it
            df_sampled = df_class.sample(n=int(desired_value), random_state=seed)
        else:
            # If the class is smaller than desired, oversample it
            df_sampled = resample(df_class, 
                                  replace=True, 
                                  n_samples=int(desired_value), 
                                  random_state=42)
        df_list.append(df_sampled)
    
    # Concatenate the list into a single DataFrame
    balanced_df = pd.concat(df_list)

    return balanced_df