import pandas as pd
from Train import plot_loss_and_acc

if __name__ == '__main__':
    dataframe = pd.read_csv('./result/train_process.csv')
    plot_loss_and_acc(dataframe)
