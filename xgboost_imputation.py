import xgboost as xgb
import pickle
import pandas as pd
import sys
import numpy as np
import math

import matplotlib.pyplot as plt


class XGBDataset:
    def __init__(self, window):
        self.window = window
        self.flattened_data = []

    def dig_holes(self):
        pass

    def imputation(self):
        pass

    def flatten_data(self):
        pass


class UCIDataset(XGBDataset):
    def __init__(self, window, source_dataset, imputing_columns, eval_masks_file):

        XGBDataset.__init__(self, window)

        self.eval_masks = np.loadtxt(eval_masks_file).astype(int)

        self.read_dataset(source_dataset)
        self.data_frame = self.data_frame.iloc[0: self.eval_masks.shape[0], :]
        self.data_frame = pd.get_dummies(self.data_frame)

        self.imputing_columns = imputing_columns

        self.evals, self.values = self.dig_holes()

        self.flattened_data = self.flatten_data()

        self.imputations = self.imputation()

    def read_dataset(self, source_dataset):
        self.data_frame = pd.read_csv(source_dataset)

    def imputation(self):
        imputations = np.transpose(np.array([[0.0] * self.evals.shape[0]] * self.evals.shape[1]))
        for i in self.imputing_columns:
            target_column = math.floor(self.window / 2) * self.evals.shape[1] + i

            training_indexes = np.where(self.eval_masks[:, i] == 0)[0].tolist()
            training_indexes = [x - math.floor(self.window / 2) for x in training_indexes]
            training_indexes = [x for x in training_indexes if -1 < x < self.flattened_data.shape[0]]

            test_indexes = np.where(self.eval_masks[:, i] == 1)[0].tolist()
            test_indexes = [x - math.floor(self.window / 2) for x in test_indexes]
            test_indexes = [x for x in test_indexes if -1 < x < self.flattened_data.shape[0]]

            training_data = self.flattened_data[training_indexes, :]
            test_data = self.flattened_data[test_indexes, :]

            training_df = pd.DataFrame(training_data)
            test_df = pd.DataFrame(test_data)

            model = xgb.XGBRegressor()
            model.fit(training_df.drop([target_column], axis=1), training_df[target_column])

            predictions = model.predict(test_df.drop([target_column], axis=1))

            for j in range(len(test_indexes)):
                imputations[test_indexes[j] + math.floor(self.window / 2)][i] = predictions[j]
            for j in range(self.evals.shape[0]):
                if imputations[j][i] == 0 and pd.notna(self.evals[j][i]):
                    imputations[j][i] = self.evals[j][i]

        return imputations

    def dig_holes(self):

        evals = self.data_frame.values

        shape = evals.shape
        evals = evals.reshape(-1)
        eval_masks = np.nonzero(self.eval_masks.reshape(-1))

        values = evals.copy()
        values[eval_masks] = np.nan

        evals = np.array(evals.reshape(shape))
        values = np.array(values.reshape(shape))

        return evals, values

    def flatten_data(self):
        flattened_data = []

        for i in range(0, self.window):
            if i == 0:
                flattened_data = self.values[0:self.values.shape[0] - self.window + 1, :]
            else:
                flattened_data = np.hstack((flattened_data, self.values[i:self.values.shape[0] - self.window + i+1, :]))

        flattened_data_df = pd.DataFrame(flattened_data)
        flattened_data_df = flattened_data_df.fillna(flattened_data_df.mean())
        flattened_data = flattened_data_df.values

        return flattened_data


class HumanActivityDataset(UCIDataset):
    def read_dataset(self, source_dataset):
        self.data_frame = pd.read_csv(source_dataset)
        self.data_frame = self.data_frame.drop(['date'], axis=1)


class EnergyDataDataset(UCIDataset):
    def read_dataset(self, source_dataset):
        self.data_frame = pd.read_csv(source_dataset)
        self.data_frame['date2'] = pd.to_datetime(self.data_frame['date'])
        self.data_frame['month'] = self.data_frame['date2'].dt.month
        self.data_frame['hour'] = self.data_frame['date2'].dt.hour
        self.data_frame = self.data_frame.drop(['date'], axis=1)
        self.data_frame = self.data_frame.drop(['date2'], axis=1)


class StockDataset(UCIDataset):
    def read_dataset(self, source_dataset):
        self.data_frame = pd.read_csv(source_dataset)
        self.data_frame = self.data_frame.drop([' dirty'], axis=1)


def count_ones_in_columns(masks):
    counts = []
    for i in range(0, masks.shape[1]):
        count = np.count_nonzero(masks[:, i])
        counts.append(count)
    return counts


dataset = UCIDataset(11, 'PRSA_data_2010.1.1-2014.12.31.csv', [5], 'all_eval_masks_air.txt')
# dataset = HumanActivityDataset(1, 'ConfLongDemo_JSI.txt', [1], 'all_eval_masks_human.txt')
# dataset = EnergyDataDataset(1, 'energydata_complete.csv', [6], 'all_eval_masks_energy.txt')
# dataset = StockDataset(11, 'stock10k.data', [1], 'all_eval_masks_stock.txt')

columns_ones = count_ones_in_columns(dataset.eval_masks)
eval_imputed_diff = np.multiply(np.abs(dataset.evals - dataset.imputations), dataset.eval_masks)

print('MAE', np.divide(np.nansum(eval_imputed_diff, axis=0), columns_ones))
print('MRE', np.divide(np.nansum(eval_imputed_diff, axis=0), np.nansum(np.multiply(np.abs(dataset.evals), dataset.eval_masks), axis=0)))

eval_imputed_diff_squared = [x ** 2 for x in eval_imputed_diff]
attributes_variance = np.nanvar(dataset.evals, axis=0)
nrms_denominator = np.multiply(attributes_variance, columns_ones)
nrms_vector = np.sqrt(np.divide(np.nansum(eval_imputed_diff_squared), nrms_denominator))
nrms_vector[nrms_vector == np.inf] = 0

print('NRMS Vector', nrms_vector)

nrms_vector_squared = [x ** 2 for x in nrms_vector]
nrms = np.sqrt(np.dot(nrms_vector_squared, columns_ones) / np.sum(columns_ones))

print('NRMS', nrms)

plt.plot(dataset.evals[:, dataset.imputing_columns[0]], 'r', label='True')
plt.plot(dataset.imputations[:, dataset.imputing_columns[0]], 'b', label='Imputed')
plt.plot(dataset.eval_masks[:, dataset.imputing_columns[0]], 'g.')
plt.subplots_adjust(wspace=2)
plt.legend(loc="upper left")
plt.title('XGB')
plt.show()






