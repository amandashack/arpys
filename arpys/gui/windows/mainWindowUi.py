from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QMenuBar


# TODO: make the window pop up in a better place on the screen
# TODO: I don't believe minimum size hint is working


class Ui_MainWindow(object):
    def setupUi(self, obj):
        obj.setMinimumSize(self.minimumSizeHint())
        obj.setObjectName("arpys")
        obj.menubar = QMenuBar()
        obj.fileMenu = obj.menubar.addMenu("file")
        obj.editMenu = obj.menubar.addMenu("edit")
        obj.helpMenu = obj.menubar.addMenu("help")
        obj.toolMenu = obj.menubar.addMenu("tools")
        obj.setMenuBar(obj.menubar)

    def minimumSizeHint(self):
        return QSize(2000, 2000)
