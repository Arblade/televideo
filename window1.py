import sys
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow
from PyQt5.QtCore import Qt
from PyQt5 import QtWidgets, uic


class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        uic.loadUi('window_principale.ui', self) # Load the .ui file
        self.show() # Show the GUI
        



app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
app.exec_()