import marshal

import pandas as pd


class DataPreprocessor(object):
    available_methods = ('fill_with_0',
                         'fill_with_value',
                         '_drop_any_na',
                         'interpolate',
                         '_drop_some + fill_with_0',
                         '_drop_some + interpolate')

    available_interpolations = ('linear', 'nearest', 'zero', 'slinear',
                                'quadratic', 'cubic')

    def __init__(self, dataframe: pd.DataFrame, imputation_method: str = None,
                 fill_value: int = -666, keep_ratio: float = None,
                 interpolation_method: str = None):
        self.version = 0.1
        self._df = dataframe

        self._log = "df"

        self._imputation_method = imputation_method
        self._keep_ratio = keep_ratio
        self._interpolation_method = interpolation_method
        self._fill_value = fill_value

    def load_df(self, df: pd.DataFrame):
        self._df = df
        # self._df = self._set_index(df)

        return self

    def load_csv(self, file: str):
        self._df = pd.read_csv(file)

        # self._df = self._set_index(df)

    @staticmethod
    def _set_index(df):
        df.set_index(['year', 'quarter'], inplace=True)
        df.sort_index(inplace=True)

        return df

    def imputate(self) -> pd.DataFrame:
        im = self._imputation_method

        assert im in self.available_methods

        if im == '_drop_any_na':
            return self._drop_any_na(self._df)

        elif im == 'interpolate':
            assert self._interpolation_method is not None
            return self._interpolate(self._df, self._interpolation_method)

        # elif im == '_drop_some + fill_with_0':
        #     assert self._keep_ratio is not None
        #     df = self._drop_some(self._df, self._keep_ratio)
        #     return self._fill_w_value(df, 0)

        elif im == '_drop_some + interpolate':
            assert self._keep_ratio is not None
            df = self._drop_some(self._df, self._keep_ratio)
            return self._interpolate(df, self._interpolation_method)

    def get_log(self):
        attrs = self.__dict__.keys()

        attrs.update({'class_code': marshal.dumps(DataPreprocessor)})

        return attrs

    # def _get_target_attrs(attr2id: dict) -> list:
    #     return [attr for attr in attr2id if 't: ' in attr]
    # ============================ IMPUTATION METHODS =================================


    def fill_w_value(self, fill_value: int):
        self._log += ".fill_w_value(%s)" % fill_value
        self._df = self._df.fillna(fill_value)
        return self

    def replace_w_value(self, original_value: int, replace_value: int):
        self._log += ".replace_w_value(to_replace=%s, value=%s)" % (
        original_value, replace_value)
        self._df = self._df.replace(to_replace=original_value,
                                    value=replace_value)
        return self


    def drop_any_na(self, axis=0):
        self._log += ".drop_any_na(how='any', axis=%s)" % axis

        self._df = self._df.dropna(how='any', axis=axis)

        return self

    @staticmethod
    def _drop_some(df_: pd.DataFrame, keep_ratio: float) -> pd.DataFrame:
        # use thresh to determine minimum number of available elements to keep column
        absolute_thresh = keep_ratio * len(df_)

        # thresh is the minimum number of NA, the 1 indicates that columns should be dropped not rows
        return df_.dropna(1, thresh=absolute_thresh)

    @staticmethod
    def _interpolate(df_: pd.DataFrame, interpolation_method: str):

        return df_.interpolate(limit_direction='both',
                               method=interpolation_method)

    def get_df(self):
        return self._df
        # def filter_dataset(data_filename: str) -> pd.DataFrame:
