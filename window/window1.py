import sys
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap
from PyQt5 import QtWidgets, uic
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QUrl

class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        ui=uic.loadUi('window_principale.ui', self) # Load the .ui file
        label = QLabel(self)
        pixmap = QPixmap('arblade.png')
        label.setPixmap(pixmap)
        url = QUrl("rtsp://192.168.8.187:1935/live/myStream")
        player = QMediaPlayer(self,QMediaPlayer.VideoSurface)
        player.setMedia(QMediaContent(url))
        videoWidget = QVideoWidget(self)
        player.setVideoOutput(videoWidget)
        player.play()
        self.show() # Show the GUI





app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
app.exec_()