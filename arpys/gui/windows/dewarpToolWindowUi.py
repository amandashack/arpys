from PyQt5.QtCore import QSize


class DewarperWindow_Ui(object):
    def setupUi(self, obj):
        obj.setMinimumSize(self.minimumSizeHint())
        obj.setObjectName("Dewarper")

    def minimumSizeHint(self):
        return (QSize(400, 800))