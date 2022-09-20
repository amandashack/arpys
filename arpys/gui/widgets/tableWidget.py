from PyQt5.QtCore import QAbstractTableModel, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableView


class TableModel(QAbstractTableModel):
    def __init__(self, context, signals, data):
        super().__init__()
        self.signals = signals
        self.context = context
        self.horizontal_labels = ['Temperature', 'fermi_kt', 'fermi_center',
                                  'fermi_amplitude', 'linear_slope', 'linear_intercept',
                                  'constant_c']
        self._data = data
        # self.setVerticalHeaderLabels(self.horizontal_labels)

    def rowCount(self, index):
        # The length of the outer list.
        return len(self._data)

    def columnCount(self, index):
        # The following takes the first sub-list, and returns
        # the length (only works if all rows are an equal length)
        return len(self._data[0])

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid():
            if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
                value = self._data[index.row()][index.column()]
                return str(value)

    def setData(self, index, value, role):
        if role == Qt.ItemDataRole.EditRole:
            self._data[index.row()][index.column()] = value
            return True
        return False

    def allData(self):
        return self._data

    def flags(self, index):
        return Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsEditable