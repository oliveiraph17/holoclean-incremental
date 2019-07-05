from dataset.table import Table, Source
from .detector import Detector


class ErrorLoaderDetector(Detector):
    """
    Detector that loads a table of constant errors with columns:
        id_col: entity ID.
        attr_col: attribute in violation.
    It can load these erros from a CSV file, a relational table, or a pandasDataFrame with the same format.
    """
    def __init__(self, fpath=None, df=None,
                 db_engine=None, table_name=None, schema_name=None,
                 id_col="_tid_", attr_col="attribute", 
                 name="ErrorLoaderDetector"):
        """
        :param fpath: (str) path to source csv file to load errors.
        :param df: (DataFrame) dataframe containing the errors.
        :param db_engine: (DBEngine) database engine object.
        :param table_name: (str) name of relational table considered for loading errors.
        :param schema_name: (str) name of schema in which :param table_name: exists.
        :param id_col: (str) ID column name.
        :param attr_col: (str) attribute column name.
        :param name: (str) name of detector.

        To load from a CSV file, :param fpath: must be specified.
        To load from a relational table, :param db_engine: and :param table_name: must be specified,
        optionally with :param schema_name:.
        """
        super(ErrorLoaderDetector, self).__init__(name)

        if fpath is not None:
            dataset_name = "errors_file"
            src = Source.FILE
        elif df is not None:
            dataset_name = "errors_df"
            src = Source.DF
        elif (db_engine is not None) and (table_name is not None):
            dataset_name = table_name
            src = Source.DB
        else:
            raise Exception("ERROR while initializing ErrorLoaderDetector: \
                            please provide (<fpath>), (<db_engine> and <table_name>), OR <df>")

        self.errors_table = Table(dataset_name, src, 
                                  exclude_attr_cols=[attr_col],
                                  fpath=fpath, df=df,
                                  schema_name=schema_name, db_engine=db_engine)

        expected_schema = [id_col, attr_col]

        if list(self.errors_table.df.columns) != expected_schema:
            raise Exception("ERROR while initializing ErrorLoaderDetector: \
                            the loaded error table does not match the expected schema {}".format(expected_schema))

        self.errors_table.df = self.errors_table.df.astype({
            id_col: int,
            attr_col: str
        })

    def setup(self, dataset=None):
        self.ds = dataset

    def detect_noisy_cells(self):
        """
        Returns a pandas.DataFrame containing loaded errors from a source.

        :return: pandas.DataFrame with columns:
            id_col: entity ID.
            attr_col: attribute in violation.
        """
        return self.errors_table.df
