from gui.widgets.tableWidget import TableModel
from PyQt5.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QWidget, QTableView
from PyQt5.QtCore import QSize


class ParamsTableWindow(QWidget):
    def __init__(self, context, signals, data):
        super(ParamsTableWindow, self).__init__()
        self.setObjectName("Parameter Editor")
        self.signals = signals
        self.context = context
        self.data = data
        self.layout = QVBoxLayout()
        self.model = TableModel(self.context, self.signals, self.data)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.bttn_save = QPushButton("Save")
        self.layout.addWidget(self.table)
        self.layout.addWidget(self.bttn_save)
        self.setMinimumSize(QSize(1000, 1000))
        self.setLayout(self.layout)
        self.make_connections()

    # def set_data(self, d):
    #    self.data = d
    #    self.model.setData(self.data)

    def make_connections(self):
        self.bttn_save.clicked.connect(self.save_data)

    def save_data(self):
        d = self.model.allData()
        self.signals.tableData.emit(d)
        self.close()
