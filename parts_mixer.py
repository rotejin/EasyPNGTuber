#!/usr/bin/env python3
"""
Parts Mixer - PNGTuberパーツ合成ツール

目と口のパーツを別々のソース画像から取得し、
4パターン（目ON/OFF x 口ON/OFF）を自動生成する。
"""
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List
from dataclasses import dataclass

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QMessageBox, QGroupBox,
    QSplitter, QScrollArea, QProgressDialog, QSpinBox, QSlider,
    QRadioButton, QButtonGroup, QComboBox, QGridLayout, QFrame, QCheckBox
)
from PySide6.QtCore import Qt, Signal, QThread, QTimer, QSettings
from PySide6.QtGui import QPixmap, QShortcut, QKeySequence

sys.path.insert(0, str(Path(__file__).parent))

from compositor import Compositor, CompositeConfig
from mask_canvas import MaskCanvas
from preview_widget import PreviewWidget
from cv2_utils import (
    load_image_as_bgra,
    save_image,
    bgra_to_qimage,
    compute_common_valid_rect,
    crop_image
)
from mask_composer import SliceAlignWorker, SliceItem


class MaskCanvasWithOverlay(MaskCanvas):
    """オーバーレイ表示機能付きマスクキャンバス

    ソース画像とベース画像を半透明で重ねて表示し、
    差分領域を視覚的に把握しやすくする。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._overlay_image: Optional[np.ndarray] = None  # ベース画像（オーバーレイ用）
        self._overlay_opacity: float = 0.5  # 0.0=ソースのみ, 1.0=ベースのみ

    def set_overlay_image(self, image: Optional[np.ndarray]):
        """オーバーレイ用ベース画像を設定"""
        if image is not None:
            self._overlay_image = image.copy()
        else:
            self._overlay_image = None
        self._update_display_pixmap()
        self.update()

    def set_overlay_opacity(self, opacity: float):
        """オーバーレイ透明度を設定（0.0〜1.0）"""
        self._overlay_opacity = max(0.0, min(1.0, opacity))
        self._update_display_pixmap()
        self.update()

    def _update_display_pixmap(self):
        """表示用ピクマップを更新（オーバーレイ合成付き）"""
        if self.base_image is None:
            return

        # オーバーレイ画像がある場合はブレンド
        if self._overlay_image is not None and self._overlay_opacity > 0:
            display_image = self._blend_with_overlay()
        else:
            display_image = self.base_image

        # QImageに変換
        qimage = bgra_to_qimage(display_image)
        self.display_pixmap = QPixmap.fromImage(qimage)

    def _blend_with_overlay(self) -> np.ndarray:
        """ソース画像とベース画像をブレンド"""
        source = self.base_image
        overlay = self._overlay_image

        # サイズが異なる場合はリサイズ
        if overlay.shape[:2] != source.shape[:2]:
            overlay = cv2.resize(overlay, (source.shape[1], source.shape[0]))

        # BGRAをBGRに変換してブレンド
        if source.shape[2] == 4:
            src_bgr = cv2.cvtColor(source, cv2.COLOR_BGRA2BGR)
            src_alpha = source[:, :, 3]
        else:
            src_bgr = source
            src_alpha = None

        if overlay.shape[2] == 4:
            ovl_bgr = cv2.cvtColor(overlay, cv2.COLOR_BGRA2BGR)
        else:
            ovl_bgr = overlay

        # ブレンド: ソース * (1-opacity) + オーバーレイ * opacity
        blended = cv2.addWeighted(
            src_bgr, 1 - self._overlay_opacity,
            ovl_bgr, self._overlay_opacity,
            0
        )

        # BGRAに戻す
        if src_alpha is not None:
            result = cv2.cvtColor(blended, cv2.COLOR_BGR2BGRA)
            result[:, :, 3] = src_alpha
            return result

        return blended


class QuadPreviewWidget(QWidget):
    """4パターン同時プレビューウィジェット"""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._previews: List[PreviewWidget] = []
        self._labels = [
            '目OFF 口OFF',
            '目ON 口OFF',
            '目OFF 口ON',
            '目ON 口ON'
        ]

        self._setup_ui()

    def _setup_ui(self):
        layout = QGridLayout(self)
        layout.setSpacing(5)

        for i, label_text in enumerate(self._labels):
            row, col = divmod(i, 2)

            container = QFrame()
            container.setFrameStyle(QFrame.StyledPanel)
            container.setStyleSheet('background-color: #2d2d30; border: 1px solid #3e3e42;')

            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(5, 5, 5, 5)

            # ラベル
            label = QLabel(label_text)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet('color: #ccc; font-weight: bold; border: none;')
            container_layout.addWidget(label)

            # プレビュー
            preview = PreviewWidget()
            preview.setMinimumSize(150, 150)
            self._previews.append(preview)
            container_layout.addWidget(preview)

            layout.addWidget(container, row, col)

    def set_images(self, images: List[Optional[np.ndarray]]):
        """4枚の画像を設定"""
        for i, preview in enumerate(self._previews):
            if i < len(images) and images[i] is not None:
                preview.set_base_image(images[i])
            else:
                preview.set_base_image(None)

    def set_scale(self, scale: float):
        """全プレビューにスケールを適用"""
        for preview in self._previews:
            preview.set_scale(scale)

    def fit_to_window(self):
        """全プレビューをウィンドウにフィット"""
        for preview in self._previews:
            preview.fit_to_window()


class PartsMixerWindow(QMainWindow):
    """Parts Mixer メインウィンドウ"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Parts Mixer - PNGTuberパーツ合成ツール')
        self.setMinimumSize(1400, 900)
        self.setAcceptDrops(True)

        # データ
        self.source_image: Optional[np.ndarray] = None
        self.source_path: str = ''
        self.items: List[SliceItem] = []
        self.current_job_id: int = 0
        self.worker: Optional[SliceAlignWorker] = None
        self.compositor = Compositor(CompositeConfig())

        # 選択インデックス
        self.base_index: int = 0
        self.eye_source_index: int = 1
        self.mouth_source_index: int = 2

        # 生成パターン
        self.generated_patterns: List[np.ndarray] = []

        # 設定保存
        self.settings = QSettings("EasyPNGTuber", "PartsMixer")
        self._last_input_dir = ""
        self._last_output_dir = ""

        # プレビュー更新デバウンス用タイマー
        self._preview_timer = QTimer()
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(200)  # 200ms デバウンス
        self._preview_timer.timeout.connect(self._do_update_previews)

        self._setup_ui()
        self._setup_shortcuts()
        self._load_settings()
        self._update_process_summary()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter = splitter

        # === 左パネル（コントロール） ===
        left_panel = QWidget()
        left_panel.setMinimumWidth(250)
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout(left_panel)

        # 分割サイズ選択
        grid_group = QGroupBox('分割サイズ')
        grid_layout = QHBoxLayout(grid_group)
        grid_layout.addWidget(QLabel('レイアウト:'))
        self.combo_grid = QComboBox()
        self.combo_grid.addItem('2x2（4枚）', 2)
        # self.combo_grid.addItem('3x3（9枚）', 3)  # Parts Mixerは2x2固定
        self.combo_grid.setCurrentIndex(0)
        self.combo_grid.currentIndexChanged.connect(self._on_grid_changed)
        grid_layout.addWidget(self.combo_grid)
        left_layout.addWidget(grid_group)

        # 画像入力
        drop_group = QGroupBox('画像入力')
        drop_layout = QVBoxLayout(drop_group)

        self.drop_zone = QLabel('表情シートをここにドロップ')
        self.drop_zone.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_zone.setStyleSheet(
            "background-color: #1e1e1e; border: 2px dashed #3e3e42; "
            "border-radius: 5px; padding: 20px; color: #888;"
        )
        self.drop_zone.setMinimumHeight(80)
        drop_layout.addWidget(self.drop_zone)

        self.btn_select = QPushButton('ファイルを選択...')
        self.btn_select.clicked.connect(self._select_file)
        drop_layout.addWidget(self.btn_select)

        self.btn_process = QPushButton('分割＆位置合わせ')
        self.btn_process.setStyleSheet(
            'background-color: #2563eb; color: white; font-weight: bold; padding: 8px;'
        )
        self.btn_process.clicked.connect(self._execute_process)
        self.btn_process.setEnabled(False)
        drop_layout.addWidget(self.btn_process)

        self.lbl_process_summary = QLabel('未処理')
        self.lbl_process_summary.setWordWrap(True)
        self.lbl_process_summary.setStyleSheet('color: #888;')
        drop_layout.addWidget(self.lbl_process_summary)

        left_layout.addWidget(drop_group)

        # 画像選択
        select_group = QGroupBox('画像選択')
        select_layout = QVBoxLayout(select_group)

        base_layout = QHBoxLayout()
        base_layout.addWidget(QLabel('ベース:'))
        self.combo_base = QComboBox()
        self.combo_base.currentIndexChanged.connect(self._on_base_changed)
        self.combo_base.setEnabled(False)
        base_layout.addWidget(self.combo_base)
        select_layout.addLayout(base_layout)

        eye_layout = QHBoxLayout()
        eye_layout.addWidget(QLabel('目ソース:'))
        self.combo_eye = QComboBox()
        self.combo_eye.currentIndexChanged.connect(self._on_eye_source_changed)
        self.combo_eye.setEnabled(False)
        eye_layout.addWidget(self.combo_eye)
        select_layout.addLayout(eye_layout)

        mouth_layout = QHBoxLayout()
        mouth_layout.addWidget(QLabel('口ソース:'))
        self.combo_mouth = QComboBox()
        self.combo_mouth.currentIndexChanged.connect(self._on_mouth_source_changed)
        self.combo_mouth.setEnabled(False)
        mouth_layout.addWidget(self.combo_mouth)
        select_layout.addLayout(mouth_layout)

        self.btn_auto_select_sources = QPushButton('低スコア候補を自動選択')
        self.btn_auto_select_sources.clicked.connect(self._auto_select_low_score_sources)
        self.btn_auto_select_sources.setEnabled(False)
        select_layout.addWidget(self.btn_auto_select_sources)

        left_layout.addWidget(select_group)

        # ブラシ設定
        brush_group = QGroupBox('ブラシ')
        brush_layout = QVBoxLayout(brush_group)

        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel('サイズ:'))
        self.spin_brush_size = QSpinBox()
        self.spin_brush_size.setRange(1, 200)
        self.spin_brush_size.setValue(30)
        self.spin_brush_size.valueChanged.connect(self._on_brush_size_changed)
        size_layout.addWidget(self.spin_brush_size)
        brush_layout.addLayout(size_layout)

        self.btn_group_mode = QButtonGroup(self)
        self.radio_add = QRadioButton('追加')
        self.radio_add.setChecked(True)
        self.radio_erase = QRadioButton('消しゴム')
        self.btn_group_mode.addButton(self.radio_add)
        self.btn_group_mode.addButton(self.radio_erase)
        brush_layout.addWidget(self.radio_add)
        brush_layout.addWidget(self.radio_erase)

        self.radio_add.toggled.connect(self._on_mode_toggled)

        # Undo/Redo
        undo_layout = QHBoxLayout()
        self.btn_undo = QPushButton('戻す')
        self.btn_undo.setToolTip(QKeySequence(QKeySequence.StandardKey.Undo).toString(QKeySequence.SequenceFormat.NativeText))
        self.btn_undo.clicked.connect(self._on_undo)
        undo_layout.addWidget(self.btn_undo)

        self.btn_redo = QPushButton('やり直し')
        self.btn_redo.setToolTip(QKeySequence(QKeySequence.StandardKey.Redo).toString(QKeySequence.SequenceFormat.NativeText))
        self.btn_redo.clicked.connect(self._on_redo)
        undo_layout.addWidget(self.btn_redo)
        brush_layout.addLayout(undo_layout)

        left_layout.addWidget(brush_group)

        # フェザー
        feather_group = QGroupBox('フェザー')
        feather_layout = QVBoxLayout(feather_group)
        feather_slider_layout = QHBoxLayout()
        feather_slider_layout.addWidget(QLabel('幅:'))
        self.slider_feather = QSlider(Qt.Horizontal)
        self.slider_feather.setRange(0, 50)
        self.slider_feather.setValue(10)
        self.slider_feather.valueChanged.connect(self._on_feather_changed)
        feather_slider_layout.addWidget(self.slider_feather)
        self.lbl_feather_value = QLabel('10px')
        feather_slider_layout.addWidget(self.lbl_feather_value)
        feather_layout.addLayout(feather_slider_layout)
        left_layout.addWidget(feather_group)

        # 保存
        save_group = QGroupBox('保存')
        save_layout = QVBoxLayout(save_group)

        self.check_auto_trim = QCheckBox('位置合わせ余白を自動トリミング')
        self.check_auto_trim.setChecked(True)
        save_layout.addWidget(self.check_auto_trim)

        trim_margin_layout = QHBoxLayout()
        trim_margin_layout.addWidget(QLabel('マージン:'))
        self.spin_trim_margin = QSpinBox()
        self.spin_trim_margin.setRange(0, 100)
        self.spin_trim_margin.setValue(0)
        self.spin_trim_margin.setSuffix('px')
        trim_margin_layout.addWidget(self.spin_trim_margin)
        trim_margin_layout.addStretch()
        save_layout.addLayout(trim_margin_layout)

        self.btn_save = QPushButton('4パターン一括保存...')
        self.btn_save.setStyleSheet('background-color: #16a34a; color: white;')
        self.btn_save.clicked.connect(self._save_all)
        self.btn_save.setEnabled(False)
        save_layout.addWidget(self.btn_save)
        left_layout.addWidget(save_group)

        left_layout.addStretch()
        splitter.addWidget(left_panel)

        # === 中央パネル（マスクキャンバス） ===
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)

        # 目パーツセクション
        eye_group = QGroupBox('目パーツ - マスク描画')
        eye_layout = QVBoxLayout(eye_group)

        eye_overlay_layout = QHBoxLayout()
        eye_overlay_layout.addWidget(QLabel('ベース透過:'))
        self.slider_eye_overlay = QSlider(Qt.Horizontal)
        self.slider_eye_overlay.setRange(0, 100)
        self.slider_eye_overlay.setValue(50)
        self.slider_eye_overlay.valueChanged.connect(self._on_eye_overlay_changed)
        eye_overlay_layout.addWidget(self.slider_eye_overlay)
        self.lbl_eye_overlay = QLabel('50%')
        self.lbl_eye_overlay.setFixedWidth(40)
        eye_overlay_layout.addWidget(self.lbl_eye_overlay)

        self.btn_clear_eye = QPushButton('クリア')
        self.btn_clear_eye.clicked.connect(self._clear_eye_mask)
        eye_overlay_layout.addWidget(self.btn_clear_eye)
        eye_layout.addLayout(eye_overlay_layout)

        eye_scroll = QScrollArea()
        eye_scroll.setWidgetResizable(False)
        eye_scroll.setMinimumHeight(200)
        self.eye_canvas = MaskCanvasWithOverlay()
        self.eye_canvas.maskChanged.connect(self._on_eye_mask_changed)
        eye_scroll.setWidget(self.eye_canvas)
        eye_layout.addWidget(eye_scroll)

        center_layout.addWidget(eye_group)

        # 口パーツセクション
        mouth_group = QGroupBox('口パーツ - マスク描画')
        mouth_layout = QVBoxLayout(mouth_group)

        mouth_overlay_layout = QHBoxLayout()
        mouth_overlay_layout.addWidget(QLabel('ベース透過:'))
        self.slider_mouth_overlay = QSlider(Qt.Horizontal)
        self.slider_mouth_overlay.setRange(0, 100)
        self.slider_mouth_overlay.setValue(50)
        self.slider_mouth_overlay.valueChanged.connect(self._on_mouth_overlay_changed)
        mouth_overlay_layout.addWidget(self.slider_mouth_overlay)
        self.lbl_mouth_overlay = QLabel('50%')
        self.lbl_mouth_overlay.setFixedWidth(40)
        mouth_overlay_layout.addWidget(self.lbl_mouth_overlay)

        self.btn_clear_mouth = QPushButton('クリア')
        self.btn_clear_mouth.clicked.connect(self._clear_mouth_mask)
        mouth_overlay_layout.addWidget(self.btn_clear_mouth)
        mouth_layout.addLayout(mouth_overlay_layout)

        mouth_scroll = QScrollArea()
        mouth_scroll.setWidgetResizable(False)
        mouth_scroll.setMinimumHeight(200)
        self.mouth_canvas = MaskCanvasWithOverlay()
        self.mouth_canvas.maskChanged.connect(self._on_mouth_mask_changed)
        mouth_scroll.setWidget(self.mouth_canvas)
        mouth_layout.addWidget(mouth_scroll)

        center_layout.addWidget(mouth_group)

        splitter.addWidget(center_panel)

        # === 右パネル（プレビュー） ===
        right_panel = QWidget()
        right_panel.setMinimumWidth(400)
        right_layout = QVBoxLayout(right_panel)

        preview_group = QGroupBox('プレビュー（4パターン）')
        preview_layout = QVBoxLayout(preview_group)

        # ズームコントロール
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel('表示:'))

        self.btn_preview_zoom_out = QPushButton('[-]')
        self.btn_preview_zoom_out.setFixedWidth(35)
        self.btn_preview_zoom_out.clicked.connect(self._preview_zoom_out)
        zoom_layout.addWidget(self.btn_preview_zoom_out)

        self.lbl_preview_zoom = QLabel('100%')
        self.lbl_preview_zoom.setFixedWidth(45)
        self.lbl_preview_zoom.setAlignment(Qt.AlignCenter)
        zoom_layout.addWidget(self.lbl_preview_zoom)

        self.btn_preview_zoom_in = QPushButton('[+]')
        self.btn_preview_zoom_in.setFixedWidth(35)
        self.btn_preview_zoom_in.clicked.connect(self._preview_zoom_in)
        zoom_layout.addWidget(self.btn_preview_zoom_in)

        self.btn_preview_fit = QPushButton('フィット')
        self.btn_preview_fit.clicked.connect(self._preview_fit)
        zoom_layout.addWidget(self.btn_preview_fit)

        zoom_layout.addStretch()
        preview_layout.addLayout(zoom_layout)

        # 4パターンプレビュー
        self.quad_preview = QuadPreviewWidget()
        preview_layout.addWidget(self.quad_preview)

        right_layout.addWidget(preview_group)
        splitter.addWidget(right_panel)

        splitter.setSizes([300, 600, 500])
        main_layout.addWidget(splitter)

        # プレビューズームレベル
        self.preview_zoom_level = 1.0

        # 最後にアクティブだったキャンバス
        self._last_active_canvas: Optional[MaskCanvasWithOverlay] = None
        self.eye_canvas.installEventFilter(self)
        self.mouth_canvas.installEventFilter(self)

    def eventFilter(self, obj, event):
        """イベントフィルタでフォーカスを追跡"""
        if event.type() == event.Type.FocusIn:
            if obj == self.eye_canvas:
                self._last_active_canvas = self.eye_canvas
            elif obj == self.mouth_canvas:
                self._last_active_canvas = self.mouth_canvas
        elif event.type() == event.Type.MouseButtonPress:
            if obj == self.eye_canvas:
                self._last_active_canvas = self.eye_canvas
            elif obj == self.mouth_canvas:
                self._last_active_canvas = self.mouth_canvas
        return super().eventFilter(obj, event)

    def _setup_shortcuts(self):
        """ショートカットキー設定"""
        self.shortcut_undo = QShortcut(QKeySequence.StandardKey.Undo, self)
        self.shortcut_undo.activated.connect(self._on_undo)
        self.shortcut_redo = QShortcut(QKeySequence.StandardKey.Redo, self)
        self.shortcut_redo.activated.connect(self._on_redo)

    # === ドラッグ&ドロップ ===

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self._load_image(path)

    def _select_file(self):
        start_dir = self._last_input_dir or ''
        path, _ = QFileDialog.getOpenFileName(
            self, '画像を選択', start_dir,
            '画像 (*.png *.jpg *.jpeg *.bmp *.webp)'
        )
        if path:
            self._load_image(path)

    def _load_image(self, path: str):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, '警告', '処理中です。')
            return

        try:
            image = load_image_as_bgra(path)
        except Exception as e:
            QMessageBox.warning(self, 'エラー', f'画像の読み込みに失敗: {e}')
            return

        h, w = image.shape[:2]
        grid_size = self.combo_grid.currentData()
        if h % grid_size != 0 or w % grid_size != 0:
            QMessageBox.warning(
                self, 'エラー',
                f'画像サイズは {grid_size}x{grid_size} で割り切れる必要があります。\n'
                f'現在: {w}x{h}'
            )
            return

        self.current_job_id += 1
        self.source_image = image
        self.source_path = path
        self._last_input_dir = str(Path(path).parent)
        self.items = []
        self.generated_patterns = []

        # サムネイル表示
        self._update_drop_zone_thumbnail(image)

        self.btn_process.setEnabled(True)
        self.btn_save.setEnabled(False)
        self._disable_combos()
        self.btn_auto_select_sources.setEnabled(False)
        self._update_process_summary()

    def _update_drop_zone_thumbnail(self, image: np.ndarray):
        """ドロップゾーンにサムネイル表示"""
        h, w = image.shape[:2]
        max_size = 60
        scale = min(max_size / w, max_size / h, 1.0)
        thumb_w, thumb_h = int(w * scale), int(h * scale)
        thumbnail = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

        qimage = bgra_to_qimage(thumbnail)
        pixmap = QPixmap.fromImage(qimage)
        self.drop_zone.setPixmap(pixmap)
        self.drop_zone.setStyleSheet(
            "background-color: #1e1e1e; border: 2px solid #4ade80; "
            "border-radius: 5px; padding: 10px;"
        )

    def _disable_combos(self):
        """コンボボックスを無効化"""
        self.combo_base.setEnabled(False)
        self.combo_eye.setEnabled(False)
        self.combo_mouth.setEnabled(False)

    def _on_grid_changed(self, index: int):
        """グリッドサイズ変更"""
        self.current_job_id += 1
        self.items = []
        self.generated_patterns = []
        self._update_combos()
        self._disable_combos()
        self.btn_auto_select_sources.setEnabled(False)

        if self.source_image is not None:
            # グリッドサイズで割り切れるか検証
            grid_size = self.combo_grid.currentData()
            h, w = self.source_image.shape[:2]
            if h % grid_size != 0 or w % grid_size != 0:
                self.btn_process.setEnabled(False)
                QMessageBox.warning(
                    self, '警告',
                    f'現在の画像サイズ ({w}x{h}) は {grid_size}x{grid_size} で割り切れません。\n'
                    f'別の画像を読み込むか、分割サイズを変更してください。'
                )
            else:
                self.btn_process.setEnabled(True)
            self.btn_save.setEnabled(False)
        self._update_process_summary()

    # === 処理実行 ===

    def _execute_process(self):
        if self.source_image is None:
            return

        self.progress = QProgressDialog('処理中...', 'キャンセル', 0, 100, self)
        self.progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress.setMinimumDuration(0)
        self.progress.setValue(0)

        grid_size = self.combo_grid.currentData()
        self.worker = SliceAlignWorker(self.source_image, self.current_job_id, grid_size)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_process_finished)
        self.worker.error.connect(self._on_process_error)
        self.worker.start()

        self.progress.canceled.connect(self._on_cancel)
        self.btn_process.setEnabled(False)
        self.combo_grid.setEnabled(False)

    def _on_progress(self, value: int, message: str):
        if hasattr(self, 'progress'):
            self.progress.setValue(value)
            self.progress.setLabelText(message)

    def _on_process_finished(self, result: tuple):
        job_id, items = result

        if job_id != self.current_job_id:
            return

        self.items = items
        self.progress.close()

        # コンボボックス更新・有効化
        self._update_combos()
        self.combo_base.setEnabled(True)
        self.combo_eye.setEnabled(True)
        self.combo_mouth.setEnabled(True)

        # キャンバス初期化
        self._update_canvases()

        self.btn_process.setEnabled(True)
        self.btn_process.setText('[OK] 処理済み（再実行）')
        self.btn_process.setStyleSheet(
            'background-color: #16a34a; color: white; font-weight: bold; padding: 8px;'
        )
        self.combo_grid.setEnabled(True)

        # プレビュー更新
        self._schedule_preview_update()
        self._update_process_summary()
        self.statusBar().showMessage('位置合わせ完了: スコアを確認し、必要なら低スコア候補を自動選択してください')

    def _on_process_error(self, message: str):
        self.progress.close()
        QMessageBox.warning(self, 'エラー', message)
        self.btn_process.setEnabled(True)
        self.combo_grid.setEnabled(True)
        self._update_process_summary()

    def _on_cancel(self):
        if self.worker:
            self.worker.requestInterruption()

    # === コンボボックス ===

    def _get_position_label(self, index: int, grid_size: int) -> str:
        """インデックスから位置ラベルを取得"""
        row = index // grid_size
        col = index % grid_size

        if grid_size == 2:
            positions = [['左上', '右上'], ['左下', '右下']]
        else:
            positions = [
                ['左上', '上', '右上'],
                ['左', '中央', '右'],
                ['左下', '下', '右下']
            ]
        return positions[row][col]

    def _format_source_label(self, index: int, grid_size: int) -> str:
        """コンボ表示ラベル（スコア付き）"""
        base_label = f'画像{index + 1}（{self._get_position_label(index, grid_size)}）'
        if index >= len(self.items):
            return base_label
        item = self.items[index]
        if item.aligned_image is None:
            return base_label
        mark = '✓' if item.alignment_success else '⚠'
        return f'{base_label} {mark}{item.alignment_score:.2f}'

    def _refresh_combo_labels(self):
        """現在の選択状態を維持したままラベルだけ更新"""
        grid_size = self.combo_grid.currentData()
        for combo in [self.combo_base, self.combo_eye, self.combo_mouth]:
            current_data = combo.currentData()
            combo.blockSignals(True)
            for i in range(combo.count()):
                combo.setItemText(i, self._format_source_label(i, grid_size))
            combo.blockSignals(False)
            if current_data is not None:
                restore_idx = combo.findData(current_data)
                if restore_idx >= 0:
                    combo.setCurrentIndex(restore_idx)

    def _update_process_summary(self):
        """処理結果サマリを更新"""
        if not hasattr(self, 'lbl_process_summary'):
            return

        if not self.items:
            self.lbl_process_summary.setText('未処理')
            self.lbl_process_summary.setStyleSheet('color: #888;')
            self.btn_auto_select_sources.setEnabled(False)
            return

        diff_items = [item for item in self.items if not item.is_base]
        if not diff_items:
            self.lbl_process_summary.setText('差分画像がありません')
            self.lbl_process_summary.setStyleSheet('color: #fbbf24;')
            self.btn_auto_select_sources.setEnabled(False)
            return

        success_count = sum(1 for item in diff_items if item.alignment_success)
        avg_score = sum(item.alignment_score for item in diff_items) / len(diff_items)
        fail_count = len(diff_items) - success_count

        if fail_count == 0:
            color = '#4ade80'
            note = '全差分の整列に成功'
        else:
            color = '#fbbf24'
            note = '低スコアがある場合は「低スコア候補を自動選択」→マスク調整推奨'

        self.lbl_process_summary.setText(
            f'整列: 成功 {success_count}/{len(diff_items)} / 平均スコア {avg_score:.2f}\n{note}'
        )
        self.lbl_process_summary.setStyleSheet(f'color: {color};')
        self.btn_auto_select_sources.setEnabled(True)

    def _auto_select_low_score_sources(self):
        """低スコア差分を目/口ソース候補として自動選択"""
        if len(self.items) < 2:
            QMessageBox.information(self, '情報', '先に分割＆位置合わせを実行してください')
            return

        candidates = [
            (idx, item.alignment_score)
            for idx, item in enumerate(self.items)
            if idx != self.base_index and item.aligned_image is not None
        ]
        if not candidates:
            QMessageBox.information(self, '情報', '候補画像がありません')
            return

        candidates.sort(key=lambda x: x[1])
        self.combo_eye.setCurrentIndex(self.combo_eye.findData(candidates[0][0]))
        if len(candidates) >= 2:
            self.combo_mouth.setCurrentIndex(self.combo_mouth.findData(candidates[1][0]))
        else:
            self.combo_mouth.setCurrentIndex(self.combo_mouth.findData(candidates[0][0]))
        self.statusBar().showMessage('低スコア候補をソース選択に反映しました')

    def _update_combos(self):
        """コンボボックスを更新"""
        grid_size = self.combo_grid.currentData()
        total = grid_size * grid_size

        for combo in [self.combo_base, self.combo_eye, self.combo_mouth]:
            combo.blockSignals(True)
            combo.clear()
            for i in range(total):
                combo.addItem(self._format_source_label(i, grid_size), i)
            combo.blockSignals(False)

        # デフォルト選択
        self.combo_base.setCurrentIndex(0)
        self.combo_eye.setCurrentIndex(min(1, total - 1))
        self.combo_mouth.setCurrentIndex(min(2, total - 1))

        self.base_index = 0
        self.eye_source_index = min(1, total - 1)
        self.mouth_source_index = min(2, total - 1)

    def _on_base_changed(self, index: int):
        if index < 0:
            return
        data = self.combo_base.currentData()
        if data is not None:
            self.base_index = data
            self._update_canvases()
            self._schedule_preview_update()

    def _on_eye_source_changed(self, index: int):
        if index < 0:
            return
        data = self.combo_eye.currentData()
        if data is not None:
            self.eye_source_index = data
            self._update_eye_canvas()
            self._schedule_preview_update()

    def _on_mouth_source_changed(self, index: int):
        if index < 0:
            return
        data = self.combo_mouth.currentData()
        if data is not None:
            self.mouth_source_index = data
            self._update_mouth_canvas()
            self._schedule_preview_update()

    # === キャンバス更新 ===

    def _update_canvases(self):
        """両キャンバスを更新"""
        self._update_eye_canvas()
        self._update_mouth_canvas()

    def _update_eye_canvas(self):
        """目キャンバスを更新"""
        if not self.items:
            return

        if self.eye_source_index < len(self.items):
            eye_image = self.items[self.eye_source_index].aligned_image
            self.eye_canvas.set_image(eye_image)

        if self.base_index < len(self.items):
            base_image = self.items[self.base_index].aligned_image
            self.eye_canvas.set_overlay_image(base_image)
            self.eye_canvas.set_overlay_opacity(self.slider_eye_overlay.value() / 100.0)

    def _update_mouth_canvas(self):
        """口キャンバスを更新"""
        if not self.items:
            return

        if self.mouth_source_index < len(self.items):
            mouth_image = self.items[self.mouth_source_index].aligned_image
            self.mouth_canvas.set_image(mouth_image)

        if self.base_index < len(self.items):
            base_image = self.items[self.base_index].aligned_image
            self.mouth_canvas.set_overlay_image(base_image)
            self.mouth_canvas.set_overlay_opacity(self.slider_mouth_overlay.value() / 100.0)

    # === ブラシ設定 ===

    def _on_brush_size_changed(self, value: int):
        self.eye_canvas.set_brush_size(value)
        self.mouth_canvas.set_brush_size(value)

    def _on_mode_toggled(self, checked: bool):
        mode = 'add' if self.radio_add.isChecked() else 'erase'
        self.eye_canvas.set_brush_mode(mode)
        self.mouth_canvas.set_brush_mode(mode)

    def _clear_eye_mask(self):
        self.eye_canvas.clear_mask()
        self._schedule_preview_update()

    def _clear_mouth_mask(self):
        self.mouth_canvas.clear_mask()
        self._schedule_preview_update()

    # === オーバーレイ透明度 ===

    def _on_eye_overlay_changed(self, value: int):
        self.lbl_eye_overlay.setText(f'{value}%')
        self.eye_canvas.set_overlay_opacity(value / 100.0)

    def _on_mouth_overlay_changed(self, value: int):
        self.lbl_mouth_overlay.setText(f'{value}%')
        self.mouth_canvas.set_overlay_opacity(value / 100.0)

    # === フェザー ===

    def _on_feather_changed(self, value: int):
        self.lbl_feather_value.setText(f'{value}px')
        self._schedule_preview_update()

    # === マスク変更 ===

    def _on_eye_mask_changed(self, mask: np.ndarray):
        self._last_active_canvas = self.eye_canvas
        self._schedule_preview_update()

    def _on_mouth_mask_changed(self, mask: np.ndarray):
        self._last_active_canvas = self.mouth_canvas
        self._schedule_preview_update()

    # === プレビュー更新（デバウンス付き） ===

    def _schedule_preview_update(self):
        """プレビュー更新をスケジュール（デバウンス）"""
        self._preview_timer.start()

    def _do_update_previews(self):
        """実際のプレビュー更新"""
        if not self.items:
            return

        eye_mask = self.eye_canvas.get_mask()
        mouth_mask = self.mouth_canvas.get_mask()

        if eye_mask is None or mouth_mask is None:
            return

        if self.base_index >= len(self.items):
            return
        if self.eye_source_index >= len(self.items):
            return
        if self.mouth_source_index >= len(self.items):
            return

        feather_width = self.slider_feather.value()
        base_image = self.items[self.base_index].aligned_image
        eye_source = self.items[self.eye_source_index].aligned_image
        mouth_source = self.items[self.mouth_source_index].aligned_image

        # 4パターン生成
        patterns = self._generate_4_patterns(
            base_image, eye_source, eye_mask,
            mouth_source, mouth_mask, feather_width
        )

        self.generated_patterns = patterns
        self.quad_preview.set_images(patterns)

        if patterns:
            self.btn_save.setEnabled(True)

    def _generate_4_patterns(
        self,
        base_image: np.ndarray,
        eye_source: np.ndarray,
        eye_mask: np.ndarray,
        mouth_source: np.ndarray,
        mouth_mask: np.ndarray,
        feather_width: int
    ) -> List[np.ndarray]:
        """4パターンを生成"""
        # マスク適用を事前計算（パフォーマンス最適化）
        masked_eye = None
        masked_mouth = None

        if eye_mask.max() > 0:
            masked_eye = self.compositor.apply_mask_to_diff(
                eye_source, eye_mask, feather_width
            )

        if mouth_mask.max() > 0:
            masked_mouth = self.compositor.apply_mask_to_diff(
                mouth_source, mouth_mask, feather_width
            )

        patterns = []

        for eye_on in [False, True]:
            for mouth_on in [False, True]:
                result = base_image.copy()

                if eye_on and masked_eye is not None:
                    result = self.compositor.composite(result, masked_eye)

                if mouth_on and masked_mouth is not None:
                    result = self.compositor.composite(result, masked_mouth)

                patterns.append(result)

        return patterns

    # === プレビューズーム ===

    def _preview_zoom_in(self):
        self.preview_zoom_level = min(3.0, self.preview_zoom_level * 1.25)
        self._apply_preview_zoom()

    def _preview_zoom_out(self):
        self.preview_zoom_level = max(0.25, self.preview_zoom_level / 1.25)
        self._apply_preview_zoom()

    def _preview_fit(self):
        self.quad_preview.fit_to_window()
        self.preview_zoom_level = 1.0
        self.lbl_preview_zoom.setText('Fit')

    def _apply_preview_zoom(self):
        self.quad_preview.set_scale(self.preview_zoom_level)
        self.lbl_preview_zoom.setText(f'{int(self.preview_zoom_level * 100)}%')

    # === Undo/Redo ===

    def _on_undo(self):
        """Undo実行（最後にアクティブだったキャンバス）"""
        canvas = self._last_active_canvas or self.eye_canvas
        if canvas.undo():
            self._schedule_preview_update()

    def _on_redo(self):
        """Redo実行（最後にアクティブだったキャンバス）"""
        canvas = self._last_active_canvas or self.eye_canvas
        if canvas.redo():
            self._schedule_preview_update()

    # === 保存 ===

    def _get_trim_rect(self) -> Optional[tuple]:
        """選択中ソースの共通有効領域からトリミング矩形を取得"""
        if not self.check_auto_trim.isChecked():
            return None
        if not self.items:
            return None

        unique_indices = {self.base_index, self.eye_source_index, self.mouth_source_index}
        valid_masks: List[np.ndarray] = []

        for idx in unique_indices:
            if idx < 0 or idx >= len(self.items):
                continue
            item = self.items[idx]
            if item.aligned_image is None:
                continue
            if item.valid_mask is not None:
                valid_masks.append(item.valid_mask)
            else:
                valid_masks.append(np.full(item.aligned_image.shape[:2], 255, dtype=np.uint8))

        if not valid_masks:
            return None

        return compute_common_valid_rect(valid_masks, margin=self.spin_trim_margin.value())

    def _save_all(self):
        if not self.generated_patterns:
            QMessageBox.warning(self, '警告', '保存する画像がありません')
            return

        output_dir = QFileDialog.getExistingDirectory(
            self, '保存先フォルダを選択', self._last_output_dir or self._last_input_dir or ''
        )
        if not output_dir:
            return
        self._last_output_dir = output_dir

        output_path = Path(output_dir)
        base_name = Path(self.source_path).stem if self.source_path else 'output'
        trim_rect = self._get_trim_rect()
        if self.check_auto_trim.isChecked() and trim_rect is None:
            QMessageBox.warning(self, '警告', 'トリミング範囲を計算できませんでした。')
            return

        # 既存ファイルチェック
        names = [
            f'{base_name}_eyeOFF_mouthOFF.png',
            f'{base_name}_eyeON_mouthOFF.png',
            f'{base_name}_eyeOFF_mouthON.png',
            f'{base_name}_eyeON_mouthON.png'
        ]

        existing = [n for n in names if (output_path / n).exists()]
        if existing:
            reply = QMessageBox.question(
                self, '確認',
                f'以下のファイルが既に存在します:\n{", ".join(existing)}\n\n上書きしますか？',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        saved = 0
        for i, (name, image) in enumerate(zip(names, self.generated_patterns)):
            try:
                output_image = crop_image(image, trim_rect) if trim_rect is not None else image
                ok = save_image(str(output_path / name), output_image)
                if not ok:
                    raise RuntimeError('save_image returned False')
                saved += 1
            except Exception as e:
                QMessageBox.warning(self, 'エラー', f'{name} の保存に失敗: {e}')

        trim_note = ''
        if trim_rect is not None:
            x, y, w, h = trim_rect
            trim_note = f'\nトリミング: x={x}, y={y}, w={w}, h={h}'

        QMessageBox.information(
            self, '完了',
            f'{saved} 枚の画像を保存しました:\n{output_dir}{trim_note}'
        )

    def _load_settings(self):
        """保存済み設定を読み込み"""
        geometry = self.settings.value('window/geometry')
        if geometry is not None:
            self.restoreGeometry(geometry)

        splitter_sizes = self.settings.value('window/splitter_sizes')
        if splitter_sizes:
            try:
                self.main_splitter.setSizes([int(v) for v in splitter_sizes])
            except Exception:
                pass

        self._last_input_dir = self.settings.value('paths/input_dir', '', type=str)
        self._last_output_dir = self.settings.value('paths/output_dir', '', type=str)

        self.spin_brush_size.setValue(self.settings.value('ui/brush_size', 30, type=int))
        self.slider_feather.setValue(self.settings.value('ui/feather', 10, type=int))
        self.slider_eye_overlay.setValue(self.settings.value('ui/eye_overlay', 50, type=int))
        self.slider_mouth_overlay.setValue(self.settings.value('ui/mouth_overlay', 50, type=int))
        self.check_auto_trim.setChecked(self.settings.value('save/auto_trim', True, type=bool))
        self.spin_trim_margin.setValue(self.settings.value('save/trim_margin', 0, type=int))

    def _save_settings(self):
        """現在設定を保存"""
        self.settings.setValue('window/geometry', self.saveGeometry())
        self.settings.setValue('window/splitter_sizes', self.main_splitter.sizes())

        self.settings.setValue('paths/input_dir', self._last_input_dir)
        self.settings.setValue('paths/output_dir', self._last_output_dir)

        self.settings.setValue('ui/brush_size', self.spin_brush_size.value())
        self.settings.setValue('ui/feather', self.slider_feather.value())
        self.settings.setValue('ui/eye_overlay', self.slider_eye_overlay.value())
        self.settings.setValue('ui/mouth_overlay', self.slider_mouth_overlay.value())
        self.settings.setValue('save/auto_trim', self.check_auto_trim.isChecked())
        self.settings.setValue('save/trim_margin', self.spin_trim_margin.value())

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName('Parts Mixer')
    app.setStyle('Fusion')

    app.setStyleSheet("""
        QMainWindow { background-color: #1e1e1e; }
        QWidget { background-color: #252526; color: #cccccc; }
        QPushButton { background-color: #0e639c; color: white; border: none; padding: 5px 15px; border-radius: 3px; }
        QPushButton:hover { background-color: #1177bb; }
        QPushButton:pressed { background-color: #094771; }
        QPushButton:disabled { background-color: #3c3c3c; color: #666; }
        QGroupBox { border: 1px solid #3e3e42; margin-top: 10px; padding-top: 10px; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
        QLabel { color: #cccccc; }
        QScrollArea { background-color: #1e1e1e; border: none; }
        QRadioButton { color: #ccc; }
        QSpinBox { background-color: #3c3c3c; border: 1px solid #3e3e42; padding: 3px; }
        QSlider::groove:horizontal { background: #3c3c3c; height: 6px; border-radius: 3px; }
        QSlider::handle:horizontal { background: #0e639c; width: 14px; margin: -4px 0; border-radius: 7px; }
        QComboBox { background-color: #3c3c3c; border: 1px solid #3e3e42; padding: 3px; }
        QComboBox::drop-down { border: none; }
        QComboBox QAbstractItemView { background-color: #3c3c3c; selection-background-color: #0e639c; }
    """)

    window = PartsMixerWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
