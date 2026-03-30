"""
マスク描画キャンバスウィジェット
"""
import cv2
import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QImage, QPixmap

from cv2_utils import bgra_to_qimage


class MaskCanvas(QWidget):
    """マスク描画キャンバス"""

    maskChanged = Signal(np.ndarray)  # マスク変更時

    def __init__(self, parent=None):
        super().__init__(parent)

        self.base_image: np.ndarray = None  # BGRA
        self.mask: np.ndarray = None        # 2D uint8 (0 or 255)
        self.display_pixmap: QPixmap = None

        # 描画設定
        self.brush_size: int = 30
        self.brush_mode: str = "add"  # "add" / "erase"
        self.drawing: bool = False
        self.last_pos: QPoint = None

        # 表示設定
        self.zoom_level: float = 1.0
        self.offset_x: float = 0.0
        self.offset_y: float = 0.0

        # マスクオーバーレイ色（半透明赤）
        self.mask_color = QColor(255, 0, 0, 128)

        # Undo/Redo履歴管理
        self._history: list = []  # マスク履歴スタック
        self._redo_stack: list = []  # Redo用スタック
        self._history_max: int = 30  # 履歴上限（動的に調整）
        self._stroke_start_mask: np.ndarray = None  # ストローク開始時のマスク

        # ブラシカーソル追跡
        self._cursor_pos: QPoint = None  # 現在のマウス位置（ウィジェット座標）

        self.setMinimumSize(200, 200)
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)  # マウス移動を常に追跡

    def set_image(self, image: np.ndarray):
        """ベース画像を設定

        Args:
            image: BGRA画像
        """
        self.base_image = image.copy()
        h, w = image.shape[:2]

        # マスクを初期化（黒=透明）
        self.mask = np.zeros((h, w), dtype=np.uint8)

        # 履歴をクリア
        self.clear_history()

        # 画像サイズに応じて履歴上限を動的調整
        self._adjust_history_limit(h, w)

        # 表示用ピクマップを更新
        self._update_display_pixmap()

        # ウィジェットサイズを設定
        self.setFixedSize(int(w * self.zoom_level), int(h * self.zoom_level))
        self.update()

    def _adjust_history_limit(self, h: int, w: int):
        """画像サイズに応じて履歴上限を動的に調整"""
        pixels = h * w
        if pixels < 1_000_000:  # 1000x1000未満
            self._history_max = 30
        elif pixels < 4_000_000:  # 2000x2000未満
            self._history_max = 20
        elif pixels < 16_000_000:  # 4000x4000未満（4K相当）
            self._history_max = 15
        else:  # 4K以上
            self._history_max = 10

    def get_mask(self) -> np.ndarray:
        """現在のマスクを取得（コピー）"""
        if self.mask is None:
            return None
        return self.mask.copy()

    def clear_mask(self):
        """マスクをクリア"""
        if self.mask is not None:
            # クリア前に履歴保存（何か描画がある場合のみ）
            if self.mask.max() > 0:
                self._push_history(self.mask.copy())
            self.mask.fill(0)
            self._update_display_pixmap()
            self.update()
            self.maskChanged.emit(self.mask.copy())

    # === Undo/Redo 履歴管理 ===

    def _push_history(self, mask_state: np.ndarray):
        """履歴にマスク状態を追加"""
        self._history.append(mask_state.copy())
        self._redo_stack.clear()  # 新しい変更でRedoスタックをクリア
        self._trim_history()

    def _trim_history(self):
        """履歴が上限を超えたら古いものを削除"""
        while len(self._history) > self._history_max:
            self._history.pop(0)

    def undo(self) -> bool:
        """Undo: 直前の状態に戻す

        Returns:
            True: Undo成功, False: 履歴がない
        """
        if not self._history or self.mask is None:
            return False
        # 描画中ならストロークをキャンセル（履歴整合性のため）
        if self.drawing:
            self._stroke_start_mask = None
            self.drawing = False
            self.last_pos = None
        # 現在の状態をRedoスタックに保存
        self._redo_stack.append(self.mask.copy())
        # 履歴から復元
        self.mask = self._history.pop()
        self._update_display_pixmap()
        self.update()
        self.maskChanged.emit(self.mask.copy())
        return True

    def redo(self) -> bool:
        """Redo: Undoした操作をやり直す

        Returns:
            True: Redo成功, False: Redoスタックが空
        """
        if not self._redo_stack or self.mask is None:
            return False
        # 描画中ならストロークをキャンセル（履歴整合性のため）
        if self.drawing:
            self._stroke_start_mask = None
            self.drawing = False
            self.last_pos = None
        # 現在の状態を履歴に保存
        self._history.append(self.mask.copy())
        # Redoスタックから復元
        self.mask = self._redo_stack.pop()
        self._update_display_pixmap()
        self.update()
        self.maskChanged.emit(self.mask.copy())
        return True

    def clear_history(self):
        """履歴をクリア（画像変更時などに呼ぶ）"""
        self._history.clear()
        self._redo_stack.clear()
        self._stroke_start_mask = None

    def _finalize_stroke(self):
        """ストロークを確定し履歴に追加"""
        if self._stroke_start_mask is not None:
            # 変更があった場合のみ履歴に追加
            if not np.array_equal(self._stroke_start_mask, self.mask):
                self._push_history(self._stroke_start_mask)
            self._stroke_start_mask = None
        self.drawing = False
        self.last_pos = None

    def set_brush_size(self, size: int):
        """ブラシサイズ設定"""
        self.brush_size = max(1, min(200, size))

    def set_brush_mode(self, mode: str):
        """ブラシモード設定（add/erase）"""
        if mode in ["add", "erase"]:
            self.brush_mode = mode

    def set_zoom(self, zoom: float):
        """ズームレベル設定"""
        self.zoom_level = max(0.1, min(5.0, zoom))
        if self.base_image is not None:
            h, w = self.base_image.shape[:2]
            self.setFixedSize(int(w * self.zoom_level), int(h * self.zoom_level))
        self.update()

    def widget_to_image_coords(self, widget_x: float, widget_y: float) -> tuple:
        """ウィジェット座標 → 画像座標"""
        image_x = int(widget_x / self.zoom_level)
        image_y = int(widget_y / self.zoom_level)
        return image_x, image_y

    def image_to_widget_coords(self, image_x: int, image_y: int) -> tuple:
        """画像座標 → ウィジェット座標"""
        widget_x = image_x * self.zoom_level
        widget_y = image_y * self.zoom_level
        return widget_x, widget_y

    def _update_display_pixmap(self):
        """表示用ピクマップを更新"""
        if self.base_image is None:
            return
        
        # ベース画像をQImageに変換
        qimage = bgra_to_qimage(self.base_image)
        self.display_pixmap = QPixmap.fromImage(qimage)

    def _draw_on_mask(self, x: int, y: int):
        """マスクに描画
        
        Args:
            x, y: 画像座標
        """
        if self.mask is None:
            return
        
        # ブラシの色
        color = 255 if self.brush_mode == "add" else 0
        
        # OpenCVで円を描画
        cv2.circle(self.mask, (x, y), self.brush_size, color, -1)

    def paintEvent(self, event):
        """描画イベント"""
        if self.base_image is None or self.display_pixmap is None:
            return

        painter = QPainter(self)

        # 背景を黒で塗りつぶし
        painter.fillRect(self.rect(), Qt.black)

        # ベース画像を描画（ズーム適用）
        scaled_pixmap = self.display_pixmap.scaled(
            int(self.base_image.shape[1] * self.zoom_level),
            int(self.base_image.shape[0] * self.zoom_level),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        scaled_pixmap.setDevicePixelRatio(self.screen().devicePixelRatio() if self.screen() else 1.0)
        painter.drawPixmap(0, 0, scaled_pixmap)

        # マスクオーバーレイを描画
        if self.mask is not None and np.any(self.mask > 0):
            mask_overlay = self._create_mask_overlay()
            painter.drawPixmap(0, 0, mask_overlay)

        # ブラシカーソルを描画
        if self._cursor_pos is not None and self.base_image is not None:
            self._draw_brush_cursor(painter)

        painter.end()

    def _draw_brush_cursor(self, painter: QPainter):
        """ブラシカーソル（円）を描画"""
        if self._cursor_pos is None:
            return

        # ブラシサイズをズーム適用
        radius = int(self.brush_size * self.zoom_level)

        # モードに応じた色
        if self.brush_mode == "add":
            color = QColor(0, 255, 0, 180)  # 緑（追加）
        else:
            color = QColor(255, 100, 100, 180)  # 赤（消しゴム）

        pen = QPen(color, 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(self._cursor_pos, radius, radius)

    def _create_mask_overlay(self) -> QPixmap:
        """マスクオーバーレイ用ピクマップを作成"""
        h, w = self.mask.shape
        
        # RGBA画像を作成（赤色半透明）
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        overlay[self.mask > 0] = [255, 0, 0, 128]  # 赤、半透明
        
        # QImageに変換
        qimage = QImage(overlay.data, w, h, w * 4, QImage.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimage.copy())
        
        # ズーム適用
        scaled_pixmap = pixmap.scaled(
            int(w * self.zoom_level),
            int(h * self.zoom_level),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        scaled_pixmap.setDevicePixelRatio(self.screen().devicePixelRatio() if self.screen() else 1.0)

        return scaled_pixmap

    def mousePressEvent(self, event):
        """マウス押下イベント"""
        if event.button() == Qt.LeftButton and self.base_image is not None:
            # ストローク開始時のマスクを保存（Undo用）
            self._stroke_start_mask = self.mask.copy()
            self.drawing = True
            self.last_pos = event.pos()

            # 画像座標に変換して描画
            img_x, img_y = self.widget_to_image_coords(event.pos().x(), event.pos().y())
            self._draw_on_mask(img_x, img_y)
            self.update()
            self.maskChanged.emit(self.mask.copy())

    def mouseMoveEvent(self, event):
        """マウス移動イベント"""
        # カーソル位置を常に更新（ブラシカーソル表示用）
        self._cursor_pos = event.pos()

        if self.drawing and self.base_image is not None:
            current_pos = event.pos()

            # 前回の位置から現在の位置まで線を描画
            last_img_x, last_img_y = self.widget_to_image_coords(
                self.last_pos.x(), self.last_pos.y()
            )
            current_img_x, current_img_y = self.widget_to_image_coords(
                current_pos.x(), current_pos.y()
            )

            # 線分上に点を補間して描画
            dist = int(np.sqrt((current_img_x - last_img_x)**2 +
                              (current_img_y - last_img_y)**2))
            # ゼロ除算回避: brush_size=1のときdivisor=0になる
            divisor = max(1, self.brush_size // 2)
            steps = max(1, dist // divisor)

            for i in range(steps + 1):
                t = i / steps
                x = int(last_img_x + (current_img_x - last_img_x) * t)
                y = int(last_img_y + (current_img_y - last_img_y) * t)
                self._draw_on_mask(x, y)

            self.last_pos = current_pos
            self.maskChanged.emit(self.mask.copy())

        # カーソル描画のために再描画（描画中も非描画中も）
        self.update()

    def mouseReleaseEvent(self, event):
        """マウス解放イベント"""
        if event.button() == Qt.LeftButton:
            # ストローク確定（履歴に追加）
            self._finalize_stroke()

    def leaveEvent(self, event):
        """マウスがウィジェットから離れた"""
        # 描画中なら履歴確定
        if self.drawing:
            self._finalize_stroke()
        # カーソル位置をクリア
        self._cursor_pos = None
        self.update()

    def enterEvent(self, event):
        """マウスがウィジェットに入った"""
        self.update()
