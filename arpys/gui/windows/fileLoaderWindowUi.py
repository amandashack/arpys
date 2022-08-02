from PyQt5.QtCore import QSize


class FileLoaderWindow_Ui(object):
    def setupUi(self, obj):
        obj.setMinimumSize(self.minimumSizeHint())
        obj.setObjectName("File Selector")

    def minimumSizeHint(self):
        return (QSize(400, 800))
