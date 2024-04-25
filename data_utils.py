import pandas as pd
from torch.utils.data import Dataset, DataLoader
"""
class GLUE_Dataset(Dataset):
    def __init__(self, 
                 data_df,
                 sentence_col, 
                 label_col) -> None:
        super().__init__()
        self.data_df = data_df
        self.sentence_col = sentence_col
        self.label_col = label_col 

    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, index):
        # Assuming the sentence is in the fourth column and the label is in the second column
        sentence = self.data_df.iloc[index, self.sentence_col]
        label = self.data_df.iloc[index, self.label_col]
        return sentence, label
"""


def get_data_df(data_dir, task, train_int): # 0 train 1 dev 2 test
    filename = 'train.tsv' if train_int == 0 else 'dev.tsv' if train_int == 1 else 'test.tsv'
    task_path = f"{data_dir}/{task}/{filename}"
    header = 0 if task == 'CoLA' else None

    df = pd.read_csv(task_path, sep='\t', header=header)
    return df


def construct_dataset(data_df, sentence_col, label_col):
    return GLUE_Dataset(data_df=data_df, sentence_col=sentence_col, label_col=label_col)

class GLUE_Dataset(Dataset):
    def __init__(self, 
                 data_df,
                 sentence_col,
                 label_col):
        super().__init__()
        self.data_df = data_df
        self.sentence_col = sentence_col
        self.label_col = label_col 

    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, index):
        sentence = self.data_df.iloc[index, self.sentence_col]
        label = self.data_df.iloc[index, self.label_col]
        return sentence, label
