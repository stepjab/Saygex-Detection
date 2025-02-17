import sys
import time
import threading
from datetime import datetime, timedelta
from ML_TRASH_VIDEO import VideoProcessor # ---------------------------------------------------------------
from huggingface_hub import hf_hub_download

from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QTimer, Signal, QObject
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QScrollArea,
    QSizePolicy
)

# -------------------------------
# Данные уведомления
# -------------------------------
class NotificationData:
    def __init__(self, nid, time_str, platform, status):
        self.nid = nid
        self.time_str = time_str
        self.platform = platform
        self.status = status  # "new" или "cleared"
        self.cleared_time = None  # время, когда мусор убран (HH:MM)

# -------------------------------
# Сигналы для обнаружения мусора
# -------------------------------
class DetectorSignals(QObject):
    garbageDetected = Signal(str, str)

# -------------------------------
# Имитация обнаружения мусора
# -------------------------------
# def simulate_detection(signals: DetectorSignals):
#     # Генерируем 4 уведомления
#     num_notifications = 5
#     for _ in range(num_notifications):
#         # Случайная задержка от 1 до 5 минут (60 до 300 секунд)
#         delay = random.uniform(1, 10)
#         time.sleep(delay)
#         # Получаем текущее время без секунд (HH:MM)
#         current_time = datetime.now().strftime("%H:%M")
#         # Выбираем случайную платформу из списка
#         platform_str = random.choice(["Платформа №1", "Платформа №2", "Платформа №3", "Платформа №4", "Платформа №8"])
#         signals.garbageDetected.emit(current_time, platform_str)

# -------------------------------
# Виджет карточки уведомления
# -------------------------------
class NotificationCard(QWidget):
    def __init__(self, data: NotificationData, parent=None):
        super().__init__(parent)
        self.data = data
        self.initUI()
        self.fadeIn()

    def initUI(self):
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setMinimumHeight(120)

        # Цвета
        self.rzdRed = "#EE3523"   # Фирменный красный
        self.grayColor = "#9E9E9E"

        # Определяем фон карточки
        bg_color = self.rzdRed if self.data.status == "new" else self.grayColor
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {bg_color};
                border-radius: 10px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # Метка времени (изначальное время обнаружения)
        self.timeLabel = QLabel(self.data.time_str)
        self.timeLabel.setAlignment(Qt.AlignCenter)
        self.timeLabel.setStyleSheet("color: white; font-size: 18px; font-weight: bold;")
        layout.addWidget(self.timeLabel)

        # Заголовок уведомления
        # Если статус уже "cleared", проверяем cleared_time
        if self.data.status == "cleared" and self.data.cleared_time:
            title_text = f"МУСОР УСПЕШНО УБРАН в {self.data.cleared_time}"
        elif self.data.status == "cleared":
            # Если почему-то cleared_time не установлено
            title_text = "МУСОР УСПЕШНО УБРАН"
        else:
            title_text = "ОБНАРУЖЕН НОВЫЙ МУСОР!"

        self.titleLabel = QLabel(title_text)
        self.titleLabel.setAlignment(Qt.AlignCenter)
        self.titleLabel.setWordWrap(True)
        self.titleLabel.setStyleSheet("color: white; font-size: 16px; font-weight: bold;")
        layout.addWidget(self.titleLabel)

        # Платформа
        self.platformLabel = QLabel(self.data.platform)
        self.platformLabel.setAlignment(Qt.AlignCenter)
        self.platformLabel.setWordWrap(True)
        self.platformLabel.setStyleSheet("color: white; font-size: 14px;")
        layout.addWidget(self.platformLabel)

        # Если статус "new" — показываем кнопку подтверждения
        if self.data.status == "new":
            self.confirmButton = QPushButton("Подтвердить уборку")
            self.confirmButton.setStyleSheet("""
                QPushButton {
                    background-color: rgba(255, 255, 255, 0.9);
                    color: #000000;
                    border: none;
                    border-radius: 5px;
                    padding: 8px 15px;
                    font-weight: bold;
                    font-size: 14px;
                    min-height: 35px;
                }
                QPushButton:hover {
                    background-color: rgba(255, 255, 255, 1.0);
                }
            """)
            self.confirmButton.clicked.connect(self.confirmCleanup)
            layout.addWidget(self.confirmButton)

    def fadeIn(self):
        from PySide6.QtCore import QPropertyAnimation
        self.setWindowOpacity(0)
        self.anim = QPropertyAnimation(self, b"windowOpacity")
        self.anim.setDuration(400)
        self.anim.setStartValue(0)
        self.anim.setEndValue(1)
        self.anim.setEasingCurve(QEasingCurve.InOutQuad)
        self.anim.start()

    def confirmCleanup(self):
        # Меняем статус, цвет карточки, скрываем кнопку
        self.data.status = "cleared"
        # Сохраняем время уборки (HH:MM)
        self.data.cleared_time = time.strftime("%H:%M")

        self.setStyleSheet("""
            QWidget {
                background-color: #9E9E9E;
                border-radius: 10px;
            }
        """)
        if hasattr(self, "confirmButton"):
            self.confirmButton.hide()

        # Обновляем заголовок с учётом времени
        self.titleLabel.setText(f"МУСОР УСПЕШНО УБРАН в {self.data.cleared_time}")
        self.repaint()

# -------------------------------
# Главное окно
# -------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Система обнаружения мусора")
        self.setMinimumSize(450, 650)
        self.initUI()

        # Изначально уведомлений нет
        self.notifications = []

        # Инициализация модели и обработчика видео
        model_path1 = "norm.pt"
        model_path2 = "yolov8m-seg.pt"
        video_path = 0  # Путь к видео ГОЙДАААААААААААААААААААААААААААААААААААААААААААААА
        self.video_processor = VideoProcessor(video_path, model_path1, model_path2)  # Ваш класс VideoProcessor
        self.video_processor.garbageDetected.connect(self.handleNewGarbage)

        # Запуск обновления времени в заголовке каждую секунду
        self.updateTimeTimer = QTimer(self)
        self.updateTimeTimer.timeout.connect(self.updateTime)
        self.updateTimeTimer.start(1000)

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ---------- Верхняя панель ----------
        header_widget = QWidget()
        header_widget.setFixedHeight(70)
        header_widget.setStyleSheet("background-color: #FFFFFF;")

        # Горизонтальный лейаут для центровки
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(10, 10, 10, 10)
        header_layout.setSpacing(10)
        header_layout.setAlignment(Qt.AlignCenter)  # выравниваем по центру

        # Логотип
        try:
            pixmap = QPixmap("assets/rzd_logo.png").scaledToHeight(24, Qt.SmoothTransformation)
            self.logo_label = QLabel()
            self.logo_label.setPixmap(pixmap)
            header_layout.addWidget(self.logo_label)
        except Exception as e:
            print("Ошибка загрузки логотипа:", e)
            self.logo_label = QLabel("RZD")

        # Заголовок
        self.titleLabel = QLabel("СИСТЕМА ОБНАРУЖЕНИЯ")
        self.titleLabel.setStyleSheet("font-size: 20px; font-weight: bold; color: #000000;")

        # Метка времени
        self.timeLabel = QLabel("")
        self.timeLabel.setStyleSheet("font-size: 16px; color: #000000;")

        # Добавляем в лейаут
        header_layout.addWidget(self.titleLabel)
        header_layout.addSpacing(15)
        header_layout.addWidget(self.timeLabel)

        main_layout.addWidget(header_widget)

        # ---------- Прокручиваемая область уведомлений ----------
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            background-color: #FFFFFF;
            QScrollBar:vertical {
                width: 0px;
                background: transparent;
            }
            QScrollBar::handle:vertical {
                background: transparent;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
                background: transparent;
            }
            QScrollBar:horizontal {
                height: 0px;
                background: transparent;
            }
            QScrollBar::handle:horizontal {
                background: transparent;
            }
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {
                width: 0px;
                background: transparent;
            }
        """)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(15, 15, 15, 15)
        self.scroll_layout.setSpacing(15)
        self.scroll_content.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_content)
        main_layout.addWidget(self.scroll_area, 1)

        # ---------- Нижняя панель (только одна кнопка) ----------
        bottom_widget = QWidget()
        bottom_widget.setFixedHeight(70)
        bottom_widget.setStyleSheet("background-color: #FFFFFF;")
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(15, 15, 15, 15)
        bottom_layout.setSpacing(15)

        self.clear_btn = QPushButton("Очистить уведомления")
        self.clear_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #EE3523;
                color: white;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover { background-color: #d62f1f; }
        """)
        self.clear_btn.clicked.connect(self.clearNotifications)
        bottom_layout.addWidget(self.clear_btn)

        self.start_button = QPushButton("Начать поиск")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
            }
            QPushButton:hover { background-color: #45a049; }
        """)
        self.start_button.clicked.connect(self.start_video_processing)
        bottom_layout.addWidget(self.start_button)

        main_layout.addWidget(bottom_widget)

    def start_video_processing(self):
        # Запускаем обработку видео в отдельном потоке
        self.video_processor.start()

    def updateTime(self):
        # Обновление метки времени (HH:MM, без секунд)
        current_time = time.strftime("%H:%M")
        self.timeLabel.setText(current_time)

    def populateNotifications(self):
        # Удаляем все виджеты из scroll_layout
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        if not self.notifications:
            empty_label = QLabel(
                "На платформах чисто!\nВ данный момент мусор не обнаружен\n\n"
                "Вы можете отслеживать ситуацию в реальном времени\n"
                "и получать уведомления, когда мусор будет найден"
            )
            empty_label.setStyleSheet("font-size: 16px; color: #000;")
            empty_label.setAlignment(Qt.AlignCenter)
            empty_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.scroll_layout.addWidget(empty_label)
        else:
            for i, notif in enumerate(self.notifications):
                card = NotificationCard(notif)
                # Эффект "отъезжания" для старых уведомлений
                if i > 0:
                    opacity = max(0.5, 1.0 - i * 0.1)
                    card.setWindowOpacity(opacity)
                    base_height = 120
                    new_height = max(60, base_height - i * 10)
                    card.setMinimumHeight(new_height)
                self.scroll_layout.addWidget(card)

        self.scroll_layout.addStretch()
        self.scroll_area.verticalScrollBar().setValue(0)

    def clearNotifications(self):
        # Отключаем сигнал, чтобы новые уведомления не добавлялись
        try:
            self.signals.garbageDetected.disconnect(self.handleNewGarbage)
        except Exception as e:
            print("Ошибка отключения сигнала:", e)
        self.notifications.clear()
        self.populateNotifications()

    def handleNewGarbage(self, time_str, platform_str):
        new_id = str(len(self.notifications) + 1)
        new_notif = NotificationData(new_id, time_str, platform_str, "new")
        self.notifications.insert(0, new_notif)
        self.populateNotifications()

def main():
    app = QApplication(sys.argv)
    # Устанавливаем шрифт FSRailway (должен быть установлен в системе)
    app.setFont(QFont("FSRailway", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
