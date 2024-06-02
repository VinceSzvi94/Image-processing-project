from PySide6.QtCore import Qt, Signal, QObject

class UpdatedImageSignal(QObject):
    image_updated = Signal()