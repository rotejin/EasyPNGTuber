"""
プレビューウィジェット
画像表示とマスク編集
"""
import cv2
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea
from PySide6.QtCore import Qt, Signal, QRect, QPoint
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QMouseEvent

from cv2_utils import convert_to_qimage, create_checkerboard


class PreviewWidget(QWidget):
    """画像プレビューウィジェット"""

    # シグナル
    mouse_pressed = Signal(int, int)  # x, y
    mouse_moved = Signal(int, int)    # x, y
    mouse_released = Signal(int, int) # x, y
    scale_changed = Signal(float)     # scale
    roi_selected = Signal(int, int, int, int)  # x, y, w, h（画像座標）

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumSize(400, 400)

        # 画像データ
        self.base_image: np.ndarray = None
        self.overlay_image: np.ndarray = None
        self.mask: np.ndarray = None

        # 表示設定
        self.show_base = True
        self.show_overlay = True
        self.show_mask = False
        self.mask_color = (0, 255, 0)  # 緑
        self.mask_alpha = 0.3

        # スケール
        self.scale = 1.0
        self.min_scale = 0.1
        self.max_scale = 5.0

        # ドラッグ
        self.dragging = False
        self.last_pos = QPoint()

        # 描画モード: "brush" or "roi_select"
        self.draw_mode = "brush"

        # ROI選択用
        self.roi_start_pos = None  # ドラッグ開始点（画像座標）
        self.roi_current_pos = None  # 現在のドラッグ位置（画像座標）
        self.roi_rect = None  # 確定したROI [x, y, w, h]
        self.roi_color = (0, 162, 232)  # 水色 (BGR)
        
        # レイアウト
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.label = QLabel("画像を選択してください")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("background-color: #2b2b2b; color: #888;")
        
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.label)
        self.scroll.setWidgetResizable(True)

        layout.addWidget(self.scroll)
        
        self.setStyleSheet("background-color: #1e1e1e;")
    
    def set_base_image(self, image: np.ndarray):
        """ベース画像を設定"""
        self.base_image = image
        self.update_display()
    
    def set_overlay_image(self, image: np.ndarray):
        """オーバーレイ画像を設定"""
        self.overlay_image = image
        self.update_display()

    def clear_overlay(self):
        """オーバーレイ画像をクリア"""
        self.overlay_image = None
        self.mask = None
        self.update_display()

    def set_mask(self, mask: np.ndarray):
        """マスクを設定"""
        self.mask = mask
        self.update_display()
    
    def set_show_base(self, show: bool):
        """ベース画像表示設定"""
        self.show_base = show
        self.update_display()
    
    def set_show_overlay(self, show: bool):
        """オーバーレイ表示設定"""
        self.show_overlay = show
        self.update_display()
    
    def set_show_mask(self, show: bool):
        """マスク表示設定"""
        self.show_mask = show
        self.update_display()

    def set_draw_mode(self, mode: str):
        """描画モード設定 ('brush' or 'roi_select')"""
        self.draw_mode = mode
        if mode == "roi_select":
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def set_roi(self, x: int, y: int, w: int, h: int):
        """ROIを設定（外部から）"""
        self.roi_rect = [x, y, w, h]
        self.update_display()

    def clear_roi(self):
        """ROIをクリア"""
        self.roi_rect = None
        self.roi_start_pos = None
        self.roi_current_pos = None
        self.update_display()
    
    def set_scale(self, scale: float):
        """スケール設定"""
        self.scale = max(self.min_scale, min(self.max_scale, scale))
        self.scale_changed.emit(self.scale)
        self.update_display()
    
    def zoom_in(self):
        """ズームイン"""
        self.set_scale(self.scale * 1.2)
    
    def zoom_out(self):
        """ズームアウト"""
        self.set_scale(self.scale / 1.2)
    
    def reset_zoom(self):
        """ズームリセット"""
        self.scale = 1.0
        self.update_display()
    
    def fit_to_window(self):
        """ウィンドウにフィット"""
        if self.base_image is None:
            return
        
        # スクロールエリアのサイズを取得
        parent = self.parent()
        while parent and not isinstance(parent, QScrollArea):
            parent = parent.parent()
        
        if parent:
            available_w = parent.viewport().width() - 20
            available_h = parent.viewport().height() - 20
        else:
            available_w = self.width() - 20
            available_h = self.height() - 20
        
        img_h, img_w = self.base_image.shape[:2]
        
        # 画像が収まるスケールを計算
        scale_w = available_w / img_w if img_w > 0 else 1.0
        scale_h = available_h / img_h if img_h > 0 else 1.0
        self.set_scale(min(scale_w, scale_h, 1.0))  # set_scale() 経由でクランプ
    
    def update_display(self):
        """表示を更新"""
        if self.base_image is None:
            self.label.setText("画像を選択してください")
            return
        
        # 表示画像を作成
        display = self._create_display_image()
        
        if display is None:
            return
        
        # スケール適用
        # Apply scale and device pixel ratio for HiDPI displays
        dpr = self._device_pixel_ratio()
        effective_scale = self.scale * dpr
        if effective_scale != 1.0:
            h, w = display.shape[:2]
            new_w = max(1, int(w * effective_scale))
            new_h = max(1, int(h * effective_scale))
            display = cv2.resize(display, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # QImageに変換
        try:
            qimage = convert_to_qimage(display)
            pixmap = QPixmap.fromImage(qimage)
            pixmap.setDevicePixelRatio(dpr)
            self.label.setPixmap(pixmap)
        except Exception as e:
            print(f"Display error: {e}")
    
    def _create_display_image(self) -> np.ndarray:
        """表示用画像を作成"""
        h, w = self.base_image.shape[:2]
        
        # ベース画像準備
        if self.show_base:
            if len(self.base_image.shape) == 3 and self.base_image.shape[2] == 4:
                # BGRA -> BGR（アルファを無視）
                base = cv2.cvtColor(self.base_image, cv2.COLOR_BGRA2BGR)
            else:
                base = self.base_image.copy()
        else:
            # チェッカーボード背景
            base = create_checkerboard((h, w))
        
        # オーバーレイ合成
        if self.show_overlay and self.overlay_image is not None:
            base = self._composite_overlay(base, self.overlay_image)
        
        # マスクオーバーレイ
        if self.show_mask and self.mask is not None:
            base = self._apply_mask_overlay(base, self.mask)

        # ROI矩形を描画
        base = self._draw_roi_overlay(base)

        return base
    
    def _composite_overlay(self, base: np.ndarray, 
                          overlay: np.ndarray) -> np.ndarray:
        """オーバーレイを合成"""
        h, w = base.shape[:2]
        
        # サイズ合わせ
        if overlay.shape[:2] != (h, w):
            overlay = cv2.resize(overlay, (w, h))
        
        # アルファチャンネル処理
        if len(overlay.shape) == 3 and overlay.shape[2] == 4:
            # アルファブレンディング
            alpha = overlay[:, :, 3].astype(float) / 255.0
            
            result = base.copy().astype(float)
            for c in range(3):
                result[:, :, c] = (
                    result[:, :, c] * (1 - alpha * 0.5) +
                    overlay[:, :, c].astype(float) * alpha * 0.5
                )
            return result.astype(np.uint8)
        else:
            # 単純ブレンド
            return cv2.addWeighted(base, 0.5, overlay, 0.5, 0)
    
    def _apply_mask_overlay(self, image: np.ndarray,
                           mask: np.ndarray) -> np.ndarray:
        """マスクオーバーレイを適用"""
        h, w = image.shape[:2]

        # マスクサイズ合わせ
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # マスクをカラー化
        mask_color = np.zeros_like(image)
        mask_color[mask > 0] = self.mask_color

        # ブレンド
        return cv2.addWeighted(image, 1.0, mask_color, self.mask_alpha, 0)

    def _draw_roi_overlay(self, image: np.ndarray) -> np.ndarray:
        """ROI矩形をオーバーレイ描画"""
        result = image.copy()

        # ドラッグ中の仮ROI
        if self.draw_mode == "roi_select" and self.roi_start_pos and self.roi_current_pos:
            x1, y1 = self.roi_start_pos
            x2, y2 = self.roi_current_pos
            # 正規化
            x, y = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            if w > 0 and h > 0:
                # 半透明の塗りつぶし
                overlay = result.copy()
                cv2.rectangle(overlay, (x, y), (x + w, y + h), self.roi_color, -1)
                result = cv2.addWeighted(overlay, 0.3, result, 0.7, 0)
                # 枠線
                cv2.rectangle(result, (x, y), (x + w, y + h), self.roi_color, 2)

        # 確定済みROI
        elif self.roi_rect:
            x, y, w, h = self.roi_rect
            # 半透明の塗りつぶし
            overlay = result.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), self.roi_color, -1)
            result = cv2.addWeighted(overlay, 0.2, result, 0.8, 0)
            # 枠線
            cv2.rectangle(result, (x, y), (x + w, y + h), self.roi_color, 2)

        return result
    
    def screen_to_image(self, x: int, y: int) -> tuple:
        """スクリーン座標を画像座標に変換"""
        # Pixmapのオフセットを計算（中央揃えの場合）
        pixmap = self.label.pixmap()
        if pixmap and not pixmap.isNull():
            offset_x = (self.label.width() - pixmap.width()) // 2
            offset_y = (self.label.height() - pixmap.height()) // 2
            x = x - max(0, offset_x)
            y = y - max(0, offset_y)

        img_x = int(x / self.scale)
        img_y = int(y / self.scale)
        return img_x, img_y
    
    def mousePressEvent(self, event: QMouseEvent):
        """マウス押下"""
        if self.base_image is None:
            return

        pos = self.label.mapFrom(self, event.pos())
        img_x, img_y = self.screen_to_image(pos.x(), pos.y())

        self.dragging = True
        self.last_pos = pos

        if self.draw_mode == "roi_select":
            # ROI選択モード：ドラッグ開始
            self.roi_start_pos = (img_x, img_y)
            self.roi_current_pos = (img_x, img_y)
        else:
            # ブラシモード
            self.mouse_pressed.emit(img_x, img_y)

    def mouseMoveEvent(self, event: QMouseEvent):
        """マウス移動"""
        if not self.dragging or self.base_image is None:
            return

        pos = self.label.mapFrom(self, event.pos())
        img_x, img_y = self.screen_to_image(pos.x(), pos.y())

        if self.draw_mode == "roi_select":
            # ROI選択モード：ドラッグ中
            self.roi_current_pos = (img_x, img_y)
            self.update_display()
        else:
            # ブラシモード
            self.mouse_moved.emit(img_x, img_y)

        self.last_pos = pos

    def mouseReleaseEvent(self, event: QMouseEvent):
        """マウス解放"""
        if not self.dragging:
            return

        pos = self.label.mapFrom(self, event.pos())
        img_x, img_y = self.screen_to_image(pos.x(), pos.y())

        self.dragging = False

        if self.draw_mode == "roi_select" and self.roi_start_pos:
            # ROI選択完了
            x1, y1 = self.roi_start_pos
            x2, y2 = img_x, img_y

            # 正規化（逆方向ドラッグ対応）
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)

            # リセット
            self.roi_start_pos = None
            self.roi_current_pos = None

            if w > 0 and h > 0:
                # ROI選択シグナル発行
                self.roi_selected.emit(x, y, w, h)
        else:
            # ブラシモード
            self.mouse_released.emit(img_x, img_y)
    
    def _device_pixel_ratio(self) -> float:
        """Get the device pixel ratio for HiDPI support."""
        screen = self.screen()
        if screen is not None:
            return screen.devicePixelRatio()
        return 1.0

    def wheelEvent(self, event):
        """ホイールイベント（ズーム）"""
        delta = event.angleDelta().y()
        
        if delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()
