#!/usr/bin/env python3
"""
Mask Composer - PNGTuber差分マスク合成ツール

2x2表情シートから顔領域をマスク描画で選択し、
ベース画像と各差分の顔を合成して出力する。
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
    QRadioButton, QButtonGroup, QTabWidget, QComboBox, QCheckBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt, Signal, QThread, QSettings
from PySide6.QtGui import QPixmap, QShortcut, QKeySequence

# パスを追加
sys.path.insert(0, str(Path(__file__).parent))

from multiprocessing import Pool, cpu_count
from aligner import Aligner, AlignConfig, _align_single_slice
from compositor import Compositor, CompositeConfig
from mask_canvas import MaskCanvas
from preview_widget import PreviewWidget
from cv2_utils import (
    load_image_as_bgra,
    save_image,
    compute_common_valid_rect,
    crop_image
)


@dataclass
class SliceItem:
    """分割画像アイテム"""
    index: int
    image: np.ndarray
    aligned_image: Optional[np.ndarray] = None
    valid_mask: Optional[np.ndarray] = None  # 255=有効領域
    alignment_success: bool = False
    alignment_score: float = 0.0
    is_base: bool = False


class SliceAlignWorker(QThread):
    """NxNスライス＆位置合わせワーカースレッド"""
    progress = Signal(int, str)  # (進捗%, メッセージ)
    finished = Signal(tuple)     # (job_id, SliceItemリスト)
    error = Signal(str)

    def __init__(self, image: np.ndarray, job_id: int, grid_size: int = 2, parent=None):
        super().__init__(parent)
        self.image = image
        self.job_id = job_id
        self.grid_size = grid_size  # 2 for 2x2, 3 for 3x3
        self.aligner = Aligner(AlignConfig())

    def run(self):
        try:
            # Step 1: Slice (NxN)
            self.progress.emit(10, f"画像を分割中（{self.grid_size}x{self.grid_size}）...")
            slices = self._slice_image_nxn(self.image, self.grid_size)

            expected_count = self.grid_size * self.grid_size
            if len(slices) != expected_count:
                self.error.emit(f"{self.grid_size}x{self.grid_size} の分割に失敗しました")
                return

            # Step 2: base_size決定
            base_size = (slices[0].shape[1], slices[0].shape[0])  # (width, height)

            # Step 3: Resize（端数ピースをbase_sizeに統一）
            self.progress.emit(20, "サイズ調整中...")
            for i in range(1, len(slices)):
                if slices[i].shape[:2] != slices[0].shape[:2]:
                    slices[i] = cv2.resize(slices[i], base_size, interpolation=cv2.INTER_LINEAR)

            # SliceItemリスト作成
            items: List[SliceItem] = []

            # ベースピース（index=0）
            base_item = SliceItem(
                index=0,
                image=slices[0],
                aligned_image=slices[0].copy(),
                valid_mask=np.full(slices[0].shape[:2], 255, dtype=np.uint8),
                alignment_success=True,
                alignment_score=1.0,
                is_base=True
            )
            items.append(base_item)

            # Step 4: Align（並列処理）
            self.progress.emit(30, "位置合わせ中（並列処理）...")

            # 位置合わせ用の引数リスト作成
            align_args = []
            for i in range(1, len(slices)):
                base_for_align = slices[0]
                target_for_align = slices[i]
                if len(base_for_align.shape) == 3 and base_for_align.shape[2] == 4:
                    base_for_align = cv2.cvtColor(base_for_align, cv2.COLOR_BGRA2BGR)
                if len(target_for_align.shape) == 3 and target_for_align.shape[2] == 4:
                    target_for_align = cv2.cvtColor(target_for_align, cv2.COLOR_BGRA2BGR)
                align_args.append((i, base_for_align, target_for_align, slices[i], base_size))

            # multiprocessing で並列実行
            n_workers = min(len(align_args), max(1, cpu_count() - 1))
            with Pool(processes=n_workers) as pool:
                results = pool.map(_align_single_slice, align_args)

            # 結果をSliceItemに変換
            for r in results:
                item = SliceItem(
                    index=r['index'],
                    image=slices[r['index']],
                    aligned_image=r['aligned_image'],
                    valid_mask=r['valid_mask'],
                    alignment_success=r['success'],
                    alignment_score=r['score'],
                )
                items.append(item)

            self.progress.emit(95, "並列処理完了")

            self.progress.emit(100, "完了")
            self.finished.emit((self.job_id, items))

        except Exception as e:
            self.error.emit(f"エラー: {str(e)}")

    def _slice_image_nxn(self, image: np.ndarray, n: int) -> List[np.ndarray]:
        """画像をNxNグリッド分割"""
        h, w = image.shape[:2]
        rows, cols = n, n

        cell_h = h // rows
        cell_w = w // cols

        # 端数計算
        extra_h = h - (cell_h * rows)
        extra_w = w - (cell_w * cols)

        slices = []
        for row in range(rows):
            for col in range(cols):
                y1 = row * cell_h
                x1 = col * cell_w

                # 最後の行/列は端数を含める
                if row == rows - 1:
                    y2 = y1 + cell_h + extra_h
                else:
                    y2 = y1 + cell_h

                if col == cols - 1:
                    x2 = x1 + cell_w + extra_w
                else:
                    x2 = x1 + cell_w

                piece = image[y1:y2, x1:x2].copy()
                slices.append(piece)

        return slices


class MaskComposerWindow(QMainWindow):
    """マスク合成メインウィンドウ"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Mask Composer - PNGTuber差分マスク合成ツール')
        self.setMinimumSize(1400, 900)
        self.setAcceptDrops(True)

        self.source_image: Optional[np.ndarray] = None
        self.source_path: str = ''
        self.items: List[SliceItem] = []
        self.mask: Optional[np.ndarray] = None
        self.current_job_id: int = 0
        self.worker: Optional[SliceAlignWorker] = None
        self.compositor = Compositor(CompositeConfig())
        self.composited_images: List[np.ndarray] = []
        self.base_index: int = 0  # 選択中のベース画像インデックス
        self._onion_opacity: float = 0.0  # オニオンスキン透明度（0.0〜1.0）

        # 設定保存
        self.settings = QSettings("EasyPNGTuber", "MaskComposer")
        self._last_input_dir = ""
        self._last_output_dir = ""

        self._setup_ui()
        self._setup_shortcuts()
        self._load_settings()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter = splitter

        # Left panel (resizable via splitter)
        left_panel = QWidget()
        left_panel.setMinimumWidth(250)
        left_panel.setMaximumWidth(450)
        left_layout = QVBoxLayout(left_panel)

        # Grid size selection
        grid_group = QGroupBox('分割サイズ')
        grid_layout = QHBoxLayout(grid_group)
        grid_layout.addWidget(QLabel('レイアウト:'))
        self.combo_grid = QComboBox()
        self.combo_grid.addItem('2x2（4枚）', 2)
        self.combo_grid.addItem('3x3（9枚）', 3)
        self.combo_grid.setCurrentIndex(0)
        self.combo_grid.currentIndexChanged.connect(self._on_grid_changed)
        grid_layout.addWidget(self.combo_grid)
        left_layout.addWidget(grid_group)

        # Drop zone
        drop_group = QGroupBox('画像入力')
        drop_layout = QVBoxLayout(drop_group)

        self.drop_zone = QLabel('2x2表情シートをここにドロップ')
        self.drop_zone.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_zone.setStyleSheet(
            "background-color: #1e1e1e; border: 2px dashed #3e3e42; "
            "border-radius: 5px; padding: 20px; color: #888;"
        )
        self.drop_zone.setMinimumHeight(100)
        drop_layout.addWidget(self.drop_zone)

        self.btn_select = QPushButton('ファイルを選択...')
        self.btn_select.clicked.connect(self._select_file)
        drop_layout.addWidget(self.btn_select)

        left_layout.addWidget(drop_group)

        # Info
        info_group = QGroupBox('情報')
        info_layout = QVBoxLayout(info_group)
        self.lbl_filename = QLabel('ファイル: 未選択')
        self.lbl_filename.setStyleSheet('color: #888;')
        info_layout.addWidget(self.lbl_filename)
        self.lbl_size = QLabel('サイズ: -')
        self.lbl_size.setStyleSheet('color: #888;')
        info_layout.addWidget(self.lbl_size)
        left_layout.addWidget(info_group)

        # Process
        process_group = QGroupBox('処理')
        process_layout = QVBoxLayout(process_group)
        self.lbl_process_status = QLabel('2x2 分割 → 位置合わせ')
        self.lbl_process_status.setStyleSheet('color: #888;')
        process_layout.addWidget(self.lbl_process_status)

        self.btn_process = QPushButton('分割＆位置合わせ')
        self.btn_process.setStyleSheet(
            'background-color: #2563eb; color: white; font-weight: bold; padding: 8px;'
        )
        self.btn_process.clicked.connect(self._execute_process)
        self.btn_process.setEnabled(False)
        process_layout.addWidget(self.btn_process)

        self.lbl_status = QLabel('画像の読み込み待ち...')
        self.lbl_status.setStyleSheet('color: #888;')
        process_layout.addWidget(self.lbl_status)

        retry_layout = QHBoxLayout()
        retry_layout.addWidget(QLabel('要調整しきい値:'))
        self.spin_issue_threshold = QDoubleSpinBox()
        self.spin_issue_threshold.setRange(0.0, 1.0)
        self.spin_issue_threshold.setSingleStep(0.05)
        self.spin_issue_threshold.setDecimals(2)
        self.spin_issue_threshold.setValue(0.6)
        self.spin_issue_threshold.valueChanged.connect(self._update_alignment_summary)
        retry_layout.addWidget(self.spin_issue_threshold)
        process_layout.addLayout(retry_layout)

        self.btn_next_issue_tab = QPushButton('次の要調整差分へ')
        self.btn_next_issue_tab.clicked.connect(self._jump_to_next_issue_tab)
        self.btn_next_issue_tab.setEnabled(False)
        process_layout.addWidget(self.btn_next_issue_tab)

        self.lbl_alignment_summary = QLabel('整列サマリ: 未実行')
        self.lbl_alignment_summary.setWordWrap(True)
        self.lbl_alignment_summary.setStyleSheet('color: #888;')
        process_layout.addWidget(self.lbl_alignment_summary)

        # 処理済みフラグ
        self.is_processed = False
        left_layout.addWidget(process_group)

        # Brush settings
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
        self.radio_add = QRadioButton('追加（顔）')
        self.radio_add.setChecked(True)
        self.radio_erase = QRadioButton('消しゴム')
        self.btn_group_mode.addButton(self.radio_add)
        self.btn_group_mode.addButton(self.radio_erase)
        brush_layout.addWidget(self.radio_add)
        brush_layout.addWidget(self.radio_erase)

        self.radio_add.toggled.connect(self._on_mode_toggled)

        self.btn_clear_mask = QPushButton('マスクをクリア')
        self.btn_clear_mask.clicked.connect(self._clear_mask)
        self.btn_clear_mask.setEnabled(False)
        brush_layout.addWidget(self.btn_clear_mask)

        # Undo/Redoボタン
        undo_redo_layout = QHBoxLayout()
        self.btn_undo = QPushButton('戻す')
        self.btn_undo.setToolTip('Ctrl+Z')
        self.btn_undo.clicked.connect(self._on_undo)
        undo_redo_layout.addWidget(self.btn_undo)

        self.btn_redo = QPushButton('やり直し')
        self.btn_redo.setToolTip('Ctrl+Y')
        self.btn_redo.clicked.connect(self._on_redo)
        undo_redo_layout.addWidget(self.btn_redo)
        brush_layout.addLayout(undo_redo_layout)

        left_layout.addWidget(brush_group)

        # Feathering
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

        # Save
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

        self.btn_save = QPushButton('全て一括保存...')
        self.btn_save.setStyleSheet('background-color: #16a34a; color: white;')
        self.btn_save.clicked.connect(self._save_all)
        self.btn_save.setEnabled(False)
        save_layout.addWidget(self.btn_save)
        left_layout.addWidget(save_group)

        left_layout.addStretch()
        splitter.addWidget(left_panel)

        # Center panel - Mask canvas
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)

        mask_group = QGroupBox('マスク描画（ベース画像の顔範囲を塗る）')
        mask_layout = QVBoxLayout(mask_group)

        # ベース画像選択
        base_select_layout = QHBoxLayout()
        base_select_layout.addWidget(QLabel('ベース画像:'))
        self.combo_base = QComboBox()
        self.combo_base.currentIndexChanged.connect(self._on_base_changed)
        self.combo_base.setEnabled(False)  # 処理完了まで無効
        base_select_layout.addWidget(self.combo_base)
        base_select_layout.addStretch()
        mask_layout.addLayout(base_select_layout)

        # Zoom controls (improved labels)
        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(QLabel('ズーム:'))

        self.btn_zoom_out = QPushButton('[-] 縮小')
        self.btn_zoom_out.setFixedWidth(80)
        self.btn_zoom_out.setToolTip('縮小')
        self.btn_zoom_out.clicked.connect(self._zoom_out)
        zoom_layout.addWidget(self.btn_zoom_out)

        self.lbl_zoom = QLabel('100%')
        self.lbl_zoom.setFixedWidth(50)
        self.lbl_zoom.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_zoom.setStyleSheet('font-weight: bold;')
        zoom_layout.addWidget(self.lbl_zoom)

        self.btn_zoom_in = QPushButton('[+] 拡大')
        self.btn_zoom_in.setFixedWidth(80)
        self.btn_zoom_in.setToolTip('拡大')
        self.btn_zoom_in.clicked.connect(self._zoom_in)
        zoom_layout.addWidget(self.btn_zoom_in)

        self.btn_zoom_fit = QPushButton('ウィンドウに合わせる')
        self.btn_zoom_fit.setToolTip('ウィンドウに合わせる')
        self.btn_zoom_fit.clicked.connect(self._zoom_fit)
        zoom_layout.addWidget(self.btn_zoom_fit)

        self.btn_zoom_100 = QPushButton('100%')
        self.btn_zoom_100.setToolTip('100%に戻す')
        self.btn_zoom_100.clicked.connect(self._zoom_reset)
        zoom_layout.addWidget(self.btn_zoom_100)

        zoom_layout.addStretch()
        mask_layout.addLayout(zoom_layout)

        # MaskCanvas
        scroll = QScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setMinimumHeight(300)

        self.mask_canvas = MaskCanvas()
        self.mask_canvas.maskChanged.connect(self._on_mask_changed)
        scroll.setWidget(self.mask_canvas)

        mask_layout.addWidget(scroll)
        center_layout.addWidget(mask_group, stretch=1)

        splitter.addWidget(center_panel)

        # Right panel - Preview (resizable via splitter)
        right_panel = QWidget()
        right_panel.setMinimumWidth(300)
        right_panel.setMaximumWidth(600)
        right_layout = QVBoxLayout(right_panel)

        preview_group = QGroupBox('プレビュー')
        preview_layout = QVBoxLayout(preview_group)

        # Preview zoom controls
        preview_zoom_layout = QHBoxLayout()
        preview_zoom_layout.addWidget(QLabel('表示サイズ:'))

        self.btn_preview_zoom_out = QPushButton('[-]')
        self.btn_preview_zoom_out.setFixedWidth(35)
        self.btn_preview_zoom_out.setToolTip('プレビュー縮小')
        self.btn_preview_zoom_out.clicked.connect(self._preview_zoom_out)
        preview_zoom_layout.addWidget(self.btn_preview_zoom_out)

        self.lbl_preview_zoom = QLabel('100%')
        self.lbl_preview_zoom.setFixedWidth(45)
        self.lbl_preview_zoom.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_zoom_layout.addWidget(self.lbl_preview_zoom)

        self.btn_preview_zoom_in = QPushButton('[+]')
        self.btn_preview_zoom_in.setFixedWidth(35)
        self.btn_preview_zoom_in.setToolTip('プレビュー拡大')
        self.btn_preview_zoom_in.clicked.connect(self._preview_zoom_in)
        preview_zoom_layout.addWidget(self.btn_preview_zoom_in)

        self.btn_preview_fit = QPushButton('ウィンドウに合わせる')
        self.btn_preview_fit.setToolTip('ウィンドウに合わせる')
        self.btn_preview_fit.clicked.connect(self._preview_fit)
        preview_zoom_layout.addWidget(self.btn_preview_fit)

        preview_zoom_layout.addStretch()
        preview_layout.addLayout(preview_zoom_layout)

        # オニオンスキン（差異確認）スライダー
        onion_layout = QHBoxLayout()
        onion_layout.addWidget(QLabel('差分比較:'))

        self.slider_onion = QSlider(Qt.Horizontal)
        self.slider_onion.setRange(0, 100)
        self.slider_onion.setValue(0)
        self.slider_onion.setToolTip('0%=合成結果、100%=差分元、中間=オニオンスキン')
        self.slider_onion.valueChanged.connect(self._on_onion_changed)
        self.slider_onion.setEnabled(False)  # 差分タブ選択時のみ有効
        onion_layout.addWidget(self.slider_onion)

        self.lbl_onion_value = QLabel('0%')
        self.lbl_onion_value.setFixedWidth(40)
        onion_layout.addWidget(self.lbl_onion_value)

        self.btn_onion_50 = QPushButton('50%')
        self.btn_onion_50.setFixedWidth(50)
        self.btn_onion_50.setToolTip('ワンクリックで50%に設定')
        self.btn_onion_50.clicked.connect(self._on_onion_50_clicked)
        self.btn_onion_50.setEnabled(False)
        onion_layout.addWidget(self.btn_onion_50)

        onion_layout.addStretch()
        preview_layout.addLayout(onion_layout)

        self.tab_preview = QTabWidget()

        # Dynamic preview widgets (will be recreated based on base selection)
        self.preview_base: Optional[PreviewWidget] = None
        self.preview_widgets: List[PreviewWidget] = []

        # ベース選択コンボの初期化とプレビュータブの構築
        self._update_base_combo()
        self._update_preview_tabs_for_base()

        # Preview zoom level tracking
        self.preview_zoom_level = 1.0

        preview_layout.addWidget(self.tab_preview)

        # 個別保存ボタン（タブの下に配置）
        self.btn_save_current = QPushButton('ベースを保存...')
        self.btn_save_current.setStyleSheet('background-color: #16a34a; color: white;')
        self.btn_save_current.clicked.connect(self._save_current)
        self.btn_save_current.setEnabled(False)
        preview_layout.addWidget(self.btn_save_current)

        # タブ切り替えシグナル接続
        self.tab_preview.currentChanged.connect(self._on_tab_changed)

        right_layout.addWidget(preview_group, stretch=1)

        splitter.addWidget(right_panel)

        splitter.setSizes([280, 700, 350])
        main_layout.addWidget(splitter)

    def _update_drop_zone_thumbnail(self, image: np.ndarray):
        """ドロップゾーンにサムネイル表示"""
        from cv2_utils import bgra_to_qimage

        # サムネイルサイズに縮小（最大80px）
        h, w = image.shape[:2]
        max_size = 80
        scale = min(max_size / w, max_size / h, 1.0)
        thumb_w, thumb_h = int(w * scale), int(h * scale)
        thumbnail = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

        # QPixmapに変換して表示
        qimage = bgra_to_qimage(thumbnail)
        pixmap = QPixmap.fromImage(qimage)
        self.drop_zone.setPixmap(pixmap)
        self.drop_zone.setStyleSheet(
            "background-color: #1e1e1e; border: 2px solid #4ade80; "
            "border-radius: 5px; padding: 10px;"
        )

    def _reset_drop_zone(self):
        """ドロップゾーンをリセット"""
        grid_size = self.combo_grid.currentData()
        self.drop_zone.clear()
        self.drop_zone.setText(f'{grid_size}x{grid_size}表情シートをここにドロップ')
        self.drop_zone.setStyleSheet(
            "background-color: #1e1e1e; border: 2px dashed #3e3e42; "
            "border-radius: 5px; padding: 20px; color: #888;"
        )

    def _on_grid_changed(self, index: int):
        """Handle grid size change"""
        grid_size = self.combo_grid.currentData()
        # サムネイルがなければテキスト更新
        if self.source_image is None:
            self.drop_zone.setText(f'{grid_size}x{grid_size}表情シートをここにドロップ')

        # ベース選択をリセット・無効化
        self._update_base_combo()
        self.combo_base.setEnabled(False)

        # プレビュータブを再構築
        self._update_preview_tabs_for_base()

        # Invalidate any running job by incrementing job_id
        self.current_job_id += 1

        # Reset state if image was loaded
        if self.source_image is not None:
            self.items = []
            self.composited_images = []
            self.is_processed = False
            self.btn_process.setText('分割＆位置合わせ')
            self.btn_process.setStyleSheet(
                'background-color: #2563eb; color: white; font-weight: bold; padding: 8px;'
            )
            self.lbl_status.setText('分割サイズを変更しました。分割＆位置合わせを実行してください。')
            self.lbl_status.setStyleSheet('color: #fbbf24;')
            self.btn_save_current.setEnabled(False)
        self._update_alignment_summary()

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
            QMessageBox.warning(self, '警告', '処理中です。しばらくお待ちください。')
            return

        try:
            image = load_image_as_bgra(path)
        except Exception as e:
            QMessageBox.warning(self, 'エラー', f'画像の読み込みに失敗しました: {e}')
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
        self.composited_images = []

        filename = Path(path).name
        self.lbl_filename.setText(f'ファイル: {filename}')
        self.lbl_filename.setStyleSheet('color: #4ade80;')
        self.lbl_size.setText(f'サイズ: {w} x {h}')
        self.lbl_size.setStyleSheet('color: #ccc;')

        # ドロップゾーンにサムネイル表示
        self._update_drop_zone_thumbnail(image)

        self.btn_process.setEnabled(True)
        self.btn_save.setEnabled(False)
        self.btn_save_current.setEnabled(False)
        self.btn_clear_mask.setEnabled(False)
        self.combo_base.setEnabled(False)
        self.base_index = 0
        self.lbl_status.setText('「分割＆位置合わせ」ボタンを押してください')
        self.lbl_status.setStyleSheet('color: #fbbf24;')  # Yellow to indicate action needed

        # ボタンを未処理状態にリセット
        self.is_processed = False
        self.btn_process.setText('[ ] 分割＆位置合わせ')
        self.btn_process.setStyleSheet(
            'background-color: #2563eb; color: white; font-weight: bold; padding: 8px;'
        )

        # キャンバスをリセット
        self.mask_canvas.set_image(np.zeros((100, 100, 4), dtype=np.uint8))
        self.preview_base.set_base_image(None)
        for pw in self.preview_widgets:
            pw.set_base_image(None)
        self._update_alignment_summary()

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
        self.combo_grid.setEnabled(False)  # Disable grid change during processing
        self.lbl_status.setText('処理中...')
        self.lbl_status.setStyleSheet('color: #60a5fa;')

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

        # ベース選択コンボを有効化・更新
        self._update_base_combo()
        self.combo_base.setEnabled(True)

        # ベース画像をキャンバスに設定（選択ベースを使用）
        base_item = items[self.base_index]
        self.mask_canvas.set_image(base_item.aligned_image)
        self.btn_clear_mask.setEnabled(True)

        # プレビュータブを構築
        self._update_preview_tabs_for_base()

        success_count = sum(1 for item in items if item.alignment_success)
        total_diffs = len(items) - 1
        self.lbl_status.setText(f'完了: {success_count - 1}/{total_diffs} 件成功')
        self.lbl_status.setStyleSheet('color: #4ade80;' if (success_count - 1) == total_diffs else 'color: #fbbf24;')
        self.btn_process.setEnabled(True)
        self.combo_grid.setEnabled(True)  # Re-enable grid change after processing

        # 処理済み状態に変更（ボタンの見た目を更新）
        self.is_processed = True
        self.btn_process.setText('[OK] 処理済み（再実行）')
        self.btn_process.setStyleSheet(
            'background-color: #16a34a; color: white; font-weight: bold; padding: 8px;'
        )

        self._update_previews()
        self._update_alignment_summary()
        self.statusBar().showMessage('処理完了: 要調整しきい値で差分を確認してください')

    def _on_process_error(self, message: str):
        self.progress.close()
        QMessageBox.warning(self, 'エラー', message)
        self.btn_process.setEnabled(True)
        self.combo_grid.setEnabled(True)  # Re-enable grid change after error
        self.lbl_status.setText('エラー')
        self.lbl_status.setStyleSheet('color: #f87171;')
        self._update_alignment_summary()

    def _on_cancel(self):
        if self.worker:
            self.worker.requestInterruption()

    def _on_brush_size_changed(self, value: int):
        self.mask_canvas.set_brush_size(value)

    def _on_mode_toggled(self, checked: bool):
        if self.radio_add.isChecked():
            self.mask_canvas.set_brush_mode('add')
        else:
            self.mask_canvas.set_brush_mode('erase')

    def _clear_mask(self):
        self.mask_canvas.clear_mask()
        self._update_previews()

    def _on_feather_changed(self, value: int):
        self.lbl_feather_value.setText(f'{value}px')
        self._update_previews()

    def _zoom_in(self):
        current_zoom = self.mask_canvas.zoom_level
        new_zoom = min(5.0, current_zoom * 1.25)
        self.mask_canvas.set_zoom(new_zoom)
        self.lbl_zoom.setText(f'{int(new_zoom * 100)}%')

    def _zoom_out(self):
        current_zoom = self.mask_canvas.zoom_level
        new_zoom = max(0.1, current_zoom / 1.25)
        self.mask_canvas.set_zoom(new_zoom)
        self.lbl_zoom.setText(f'{int(new_zoom * 100)}%')

    def _zoom_fit(self):
        if self.source_image is None:
            return
        h, w = self.source_image.shape[:2]
        # グリッドサイズに応じてスライスサイズを計算
        grid_size = self.combo_grid.currentData()
        slice_w = w // grid_size
        slice_h = h // grid_size
        available_width = 600
        available_height = 500
        scale = min(available_width / slice_w, available_height / slice_h, 1.0)
        self.mask_canvas.set_zoom(scale)
        self.lbl_zoom.setText(f'{int(scale * 100)}%')

    def _zoom_reset(self):
        """Reset mask canvas zoom to 100%"""
        self.mask_canvas.set_zoom(1.0)
        self.lbl_zoom.setText('100%')

    def _preview_zoom_in(self):
        """Enlarge all preview widgets"""
        self.preview_zoom_level = min(3.0, self.preview_zoom_level * 1.25)
        self._apply_preview_zoom()

    def _preview_zoom_out(self):
        """Shrink all preview widgets"""
        self.preview_zoom_level = max(0.25, self.preview_zoom_level / 1.25)
        self._apply_preview_zoom()

    def _preview_fit(self):
        """Fit all previews to window"""
        all_previews = [self.preview_base] + self.preview_widgets
        for pw in all_previews:
            pw.fit_to_window()
        # Sync internal zoom level with actual preview scale
        # Use base preview's scale as reference (all should be same)
        if self.preview_base.base_image is not None:
            self.preview_zoom_level = self.preview_base.scale
            self.lbl_preview_zoom.setText(f'{int(self.preview_zoom_level * 100)}%')
        else:
            self.preview_zoom_level = 1.0
            self.lbl_preview_zoom.setText('100%')

    def _apply_preview_zoom(self):
        """Apply zoom level to all preview widgets"""
        if not hasattr(self, 'preview_zoom_level') or not hasattr(self, 'lbl_preview_zoom'):
            return
        if self.preview_base is None:
            return
        all_previews = [self.preview_base] + self.preview_widgets
        for pw in all_previews:
            pw.set_scale(self.preview_zoom_level)
        self.lbl_preview_zoom.setText(f'{int(self.preview_zoom_level * 100)}%')

    def _on_mask_changed(self, mask: np.ndarray):
        self.mask = mask
        self._update_previews()

    def _on_tab_changed(self, index: int):
        """タブ切り替え時にボタンラベルを更新"""
        if not hasattr(self, 'btn_save_current'):
            return
        tab_text = self.tab_preview.tabText(index)
        self.btn_save_current.setText(f'{tab_text}を保存...')

        # オニオンスキンスライダーの有効/無効切替
        is_diff_tab = (index > 0)  # index=0はベースタブ
        self.slider_onion.setEnabled(is_diff_tab)
        self.btn_onion_50.setEnabled(is_diff_tab)
        if not is_diff_tab:
            # ベースタブ: スライダーを0%にリセット
            self.slider_onion.blockSignals(True)
            self.slider_onion.setValue(0)
            self.lbl_onion_value.setText('0%')
            self._onion_opacity = 0.0
            self.slider_onion.blockSignals(False)
        else:
            # 差分タブ: 現在のスライダー値でプレビューを更新
            # （タブ切替時にプレビュー画像を同期させる）
            self._update_current_onion_preview()

    def _get_position_label(self, index: int, grid_size: int) -> str:
        """インデックスから位置ラベルを取得"""
        row = index // grid_size
        col = index % grid_size

        if grid_size == 2:
            positions = [['左上', '右上'], ['左下', '右下']]
        else:  # 3x3
            positions = [
                ['左上', '上', '右上'],
                ['左', '中央', '右'],
                ['左下', '下', '右下']
            ]
        return positions[row][col]

    def _update_base_combo(self):
        """グリッドサイズに応じてベース選択コンボボックスを更新"""
        self.combo_base.blockSignals(True)
        self.combo_base.clear()

        grid_size = self.combo_grid.currentData()
        total = grid_size * grid_size

        for i in range(total):
            label = self._get_position_label(i, grid_size)
            self.combo_base.addItem(f'画像{i + 1}（{label}）', i)

        self.combo_base.setCurrentIndex(0)
        self.base_index = 0
        self.combo_base.blockSignals(False)

    def _on_base_changed(self, index: int):
        """ベース画像変更時"""
        if index < 0:
            return

        data = self.combo_base.currentData()
        if data is None:
            return

        # マスクが描画済みの場合は確認ダイアログを表示
        current_mask = self.mask_canvas.get_mask()
        if current_mask is not None and current_mask.max() > 0:
            reply = QMessageBox.question(
                self, '確認',
                'ベース画像を変更すると、描画中のマスクがクリアされます。\n続行しますか？',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                # キャンセル：コンボボックスを元に戻す
                self.combo_base.blockSignals(True)
                self.combo_base.setCurrentIndex(self.base_index)
                self.combo_base.blockSignals(False)
                return

        self.base_index = data

        # マスクキャンバスの背景画像を更新
        if self.items and self.base_index < len(self.items):
            base_item = self.items[self.base_index]
            self.mask_canvas.set_image(base_item.aligned_image)

        # プレビュータブを再構築
        self._update_preview_tabs_for_base()

        # プレビュー更新
        self._update_previews()
        self._update_alignment_summary()

    def _update_preview_tabs_for_base(self):
        """選択ベースに応じてプレビュータブを再構築"""
        # 既存タブをクリア
        while self.tab_preview.count() > 0:
            widget = self.tab_preview.widget(0)
            self.tab_preview.removeTab(0)
            widget.deleteLater()
        self.preview_widgets.clear()

        # オニオンスキン状態をリセット
        self._onion_opacity = 0.0
        if hasattr(self, 'slider_onion'):
            self.slider_onion.blockSignals(True)
            self.slider_onion.setValue(0)
            self.lbl_onion_value.setText('0%')
            self.slider_onion.blockSignals(False)

        grid_size = self.combo_grid.currentData()
        total = grid_size * grid_size

        # ベースタブを追加
        base_label = self._get_position_label(self.base_index, grid_size)
        self.preview_base = PreviewWidget()
        self.tab_preview.addTab(self.preview_base, f'ベース（{base_label}）')

        # 差分タブを追加（ベース以外の画像）
        diff_indices = self._get_diff_indices()
        for i, diff_idx in enumerate(diff_indices):
            preview = PreviewWidget()
            pos_label = self._get_position_label(diff_idx, grid_size)
            self.tab_preview.addTab(preview, f'差分{i + 1}（{pos_label}）')
            self.preview_widgets.append(preview)

        # 個別保存ボタンのラベルを同期
        self._on_tab_changed(self.tab_preview.currentIndex())

        # プレビューズーム状態を引き継ぐ
        self._apply_preview_zoom()

    def _get_diff_indices(self) -> List[int]:
        """現在のベース以外の画像インデックスを取得"""
        grid_size = self.combo_grid.currentData()
        total = grid_size * grid_size
        return [i for i in range(total) if i != self.base_index]

    def _collect_issue_diff_indices(self) -> List[int]:
        """低スコア/失敗の差分インデックスを収集"""
        threshold = self.spin_issue_threshold.value() if hasattr(self, 'spin_issue_threshold') else 0.6
        issue_indices: List[int] = []
        for idx in self._get_diff_indices():
            if idx >= len(self.items):
                continue
            item = self.items[idx]
            if item.aligned_image is None:
                issue_indices.append(idx)
                continue
            if (not item.alignment_success) or item.alignment_score < threshold:
                issue_indices.append(idx)
        return issue_indices

    def _jump_to_next_issue_tab(self):
        """次の要調整差分タブへ移動"""
        issue_indices = self._collect_issue_diff_indices()
        if not issue_indices:
            QMessageBox.information(self, '情報', '要調整の差分はありません')
            return

        diff_indices = self._get_diff_indices()
        issue_tab_indices = [diff_indices.index(idx) + 1 for idx in issue_indices if idx in diff_indices]
        if not issue_tab_indices:
            QMessageBox.information(self, '情報', '要調整差分タブが見つかりません')
            return

        current_tab = self.tab_preview.currentIndex()
        next_tab = None
        for tab_idx in issue_tab_indices:
            if tab_idx > current_tab:
                next_tab = tab_idx
                break
        if next_tab is None:
            next_tab = issue_tab_indices[0]

        self.tab_preview.setCurrentIndex(next_tab)
        self.statusBar().showMessage('要調整差分タブへ移動しました')

    def _update_alignment_summary(self, *_):
        """整列サマリ表示を更新"""
        if not hasattr(self, 'lbl_alignment_summary'):
            return

        if not self.items:
            self.lbl_alignment_summary.setText('整列サマリ: 未実行')
            self.lbl_alignment_summary.setStyleSheet('color: #888;')
            if hasattr(self, 'btn_next_issue_tab'):
                self.btn_next_issue_tab.setEnabled(False)
            return

        diff_indices = self._get_diff_indices()
        if not diff_indices:
            self.lbl_alignment_summary.setText('整列サマリ: 差分なし')
            self.lbl_alignment_summary.setStyleSheet('color: #888;')
            if hasattr(self, 'btn_next_issue_tab'):
                self.btn_next_issue_tab.setEnabled(False)
            return

        diff_items = [self.items[idx] for idx in diff_indices if idx < len(self.items)]
        if not diff_items:
            self.lbl_alignment_summary.setText('整列サマリ: 差分なし')
            self.lbl_alignment_summary.setStyleSheet('color: #888;')
            if hasattr(self, 'btn_next_issue_tab'):
                self.btn_next_issue_tab.setEnabled(False)
            return

        success_count = sum(1 for item in diff_items if item.alignment_success)
        avg_score = sum(item.alignment_score for item in diff_items) / len(diff_items)
        issue_count = len(self._collect_issue_diff_indices())
        threshold = self.spin_issue_threshold.value() if hasattr(self, 'spin_issue_threshold') else 0.6

        if issue_count == 0:
            color = '#4ade80'
            note = '良好'
        else:
            color = '#fbbf24'
            note = '要調整あり'

        self.lbl_alignment_summary.setText(
            f'整列サマリ: 成功 {success_count}/{len(diff_items)} / 要調整 {issue_count}\n'
            f'平均スコア {avg_score:.2f}（しきい値 {threshold:.2f}） {note}'
        )
        self.lbl_alignment_summary.setStyleSheet(f'color: {color};')
        if hasattr(self, 'btn_next_issue_tab'):
            self.btn_next_issue_tab.setEnabled(issue_count > 0)

    def _update_previews(self):
        if not self.items:
            return

        mask = self.mask_canvas.get_mask()
        if mask is None:
            return

        # ベース画像を動的に取得
        if self.base_index >= len(self.items):
            return
        feather_width = self.slider_feather.value()
        base_image = self.items[self.base_index].aligned_image

        self.preview_base.set_base_image(base_image)

        self.composited_images = []

        # 差分インデックスとpreview_widgetsは同じ順序で対応
        diff_indices = self._get_diff_indices()
        current_tab = self.tab_preview.currentIndex()

        for i, preview_widget in enumerate(self.preview_widgets):
            if i < len(diff_indices):
                diff_idx = diff_indices[i]
                if diff_idx < len(self.items):
                    diff_image = self.items[diff_idx].aligned_image
                    masked_diff = self.compositor.apply_mask_to_diff(
                        diff_image, mask, feather_width
                    )
                    composited = self.compositor.composite(base_image, masked_diff)
                    self.composited_images.append(composited)

                    # 現在のタブのみオニオンスキン適用
                    tab_index = i + 1  # タブインデックス（ベースが0）
                    if tab_index == current_tab and self._onion_opacity > 0:
                        # オニオンスキン表示
                        display = self._blend_onion_skin(composited, diff_image)
                        preview_widget.set_base_image(display)
                    else:
                        # 合成結果を表示
                        preview_widget.set_base_image(composited)
                else:
                    preview_widget.set_base_image(None)
            else:
                preview_widget.set_base_image(None)

        if self.composited_images:
            self.btn_save.setEnabled(True)
            self.btn_save_current.setEnabled(True)

    def _blend_onion_skin(self, composited: np.ndarray, diff_image: np.ndarray) -> np.ndarray:
        """オニオンスキン: 合成結果と差分元をブレンド"""
        if self._onion_opacity <= 0:
            return composited
        if self._onion_opacity >= 1.0:
            return diff_image

        # BGRAをBGRに変換してブレンド
        has_alpha = (len(composited.shape) == 3 and composited.shape[2] == 4)

        if has_alpha:
            comp_bgr = cv2.cvtColor(composited, cv2.COLOR_BGRA2BGR)
            comp_alpha = composited[:, :, 3]
        else:
            comp_bgr = composited
            comp_alpha = None

        if len(diff_image.shape) == 3 and diff_image.shape[2] == 4:
            diff_bgr = cv2.cvtColor(diff_image, cv2.COLOR_BGRA2BGR)
        else:
            diff_bgr = diff_image

        blended = cv2.addWeighted(
            comp_bgr, 1 - self._onion_opacity,
            diff_bgr, self._onion_opacity,
            0
        )

        # 元がBGRAならアルファチャンネルを復元してBGRAで返す
        if has_alpha and comp_alpha is not None:
            blended_bgra = cv2.cvtColor(blended, cv2.COLOR_BGR2BGRA)
            blended_bgra[:, :, 3] = comp_alpha
            return blended_bgra

        return blended

    def _get_trim_rect(self, indices: List[int]) -> Optional[tuple]:
        """指定インデックス群の共通有効領域からトリミング矩形を算出"""
        if not self.check_auto_trim.isChecked():
            return None

        valid_masks = []
        for idx in indices:
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

    def _save_current(self):
        """現在選択中のタブの画像を個別保存"""
        current_index = self.tab_preview.currentIndex()
        base_name = Path(self.source_path).stem if self.source_path else 'output'
        grid_size = self.combo_grid.currentData()

        # 画像の存在チェックと取得
        if current_index == 0:
            # ベース画像を保存（選択ベースを使用）
            if self.base_index >= len(self.items) or self.items[self.base_index].aligned_image is None:
                QMessageBox.warning(self, '警告', '保存する画像がありません')
                return
            image = self.items[self.base_index].aligned_image
            base_label = self._get_position_label(self.base_index, grid_size)
            default_name = f'{base_name}_base_{base_label}.png'
            trim_indices = [self.base_index]
        else:
            # 差分画像を保存
            comp_index = current_index - 1
            if comp_index >= len(self.composited_images):
                QMessageBox.warning(self, '警告', '保存する画像がありません')
                return
            image = self.composited_images[comp_index]
            diff_indices = self._get_diff_indices()
            if comp_index < len(diff_indices):
                diff_idx = diff_indices[comp_index]
                pos_label = self._get_position_label(diff_idx, grid_size)
                default_name = f'{base_name}_差分{current_index}_{pos_label}.png'
                trim_indices = [self.base_index, diff_idx]
            else:
                default_name = f'{base_name}_差分{current_index}.png'
                trim_indices = [self.base_index]

        trim_rect = self._get_trim_rect(trim_indices)
        if self.check_auto_trim.isChecked() and trim_rect is None:
            QMessageBox.warning(self, '警告', 'トリミング範囲を計算できませんでした。画像サイズや位置合わせ結果を確認してください。')
            return

        # ファイル保存ダイアログ
        default_dir = self._last_output_dir or (str(Path(self.source_path).parent) if self.source_path else '')
        path, _ = QFileDialog.getSaveFileName(
            self, '画像を保存',
            str(Path(default_dir) / default_name),
            '画像 (*.png)'
        )
        if path:
            # 拡張子がなければ追加
            if not path.lower().endswith('.png'):
                path += '.png'
            try:
                output_image = crop_image(image, trim_rect) if trim_rect is not None else image
                ok = save_image(path, output_image)
                if not ok:
                    raise RuntimeError('save_image returned False')
                self._last_output_dir = str(Path(path).parent)
                QMessageBox.information(self, '完了', f'保存しました:\n{path}')
            except Exception as e:
                QMessageBox.warning(self, 'エラー', f'保存に失敗しました:\n{e}')

    def _save_all(self):
        if not self.items or not self.composited_images:
            QMessageBox.warning(self, '警告', '保存できる画像がありません')
            return

        output_dir = QFileDialog.getExistingDirectory(
            self, '保存先フォルダを選択', self._last_output_dir or self._last_input_dir or ''
        )
        if not output_dir:
            return
        self._last_output_dir = output_dir

        output_path = Path(output_dir)
        base_name = Path(self.source_path).stem if self.source_path else 'output'
        grid_size = self.combo_grid.currentData()
        diff_indices = self._get_diff_indices()
        trim_indices = [self.base_index, *diff_indices]
        trim_rect = self._get_trim_rect(trim_indices)

        if self.check_auto_trim.isChecked() and trim_rect is None:
            QMessageBox.warning(self, '警告', 'トリミング範囲を計算できませんでした。画像サイズや位置合わせ結果を確認してください。')
            return

        saved = 0

        # ベース画像を保存（選択ベースを使用）
        base_image = self.items[self.base_index].aligned_image
        base_label = self._get_position_label(self.base_index, grid_size)
        filename = f'{base_name}_01_base_{base_label}.png'
        base_output = crop_image(base_image, trim_rect) if trim_rect is not None else base_image
        if not save_image(str(output_path / filename), base_output):
            QMessageBox.warning(self, 'エラー', f'{filename} の保存に失敗しました')
            return
        saved += 1

        # 合成画像を保存
        for i, image in enumerate(self.composited_images):
            if i < len(diff_indices):
                diff_idx = diff_indices[i]
                pos_label = self._get_position_label(diff_idx, grid_size)
                filename = f'{base_name}_{i + 2:02d}_merged_{pos_label}.png'
            else:
                filename = f'{base_name}_{i + 2:02d}_merged.png'
            output_image = crop_image(image, trim_rect) if trim_rect is not None else image
            if not save_image(str(output_path / filename), output_image):
                QMessageBox.warning(self, 'エラー', f'{filename} の保存に失敗しました')
                return
            saved += 1

        trim_note = ''
        if trim_rect is not None:
            x, y, w, h = trim_rect
            trim_note = f'\nトリミング: x={x}, y={y}, w={w}, h={h}'

        QMessageBox.information(
            self, '完了',
            f'{saved} 枚の画像を保存しました:\n{output_dir}{trim_note}'
        )

    # === ショートカットキー設定 ===

    def _setup_shortcuts(self):
        """ショートカットキーを設定"""
        self.shortcut_undo = QShortcut(QKeySequence.StandardKey.Undo, self)
        self.shortcut_undo.activated.connect(self._on_undo)
        self.shortcut_redo = QShortcut(QKeySequence.StandardKey.Redo, self)
        self.shortcut_redo.activated.connect(self._on_redo)

    def _on_undo(self):
        """Undo実行"""
        if self.mask_canvas.undo():
            self._update_previews()

    def _on_redo(self):
        """Redo実行"""
        if self.mask_canvas.redo():
            self._update_previews()

    # === オニオンスキン（差異確認）機能 ===

    def _on_onion_changed(self, value: int):
        """オニオンスキンスライダー変更時"""
        self._onion_opacity = value / 100.0
        self.lbl_onion_value.setText(f'{value}%')
        self._update_current_onion_preview()

    def _on_onion_50_clicked(self):
        """50%ボタン押下時"""
        self.slider_onion.setValue(50)

    def _update_current_onion_preview(self):
        """現在の差分タブのプレビューをオニオンスキンで更新"""
        current_index = self.tab_preview.currentIndex()
        if current_index <= 0:
            return

        widget_index = current_index - 1
        if widget_index >= len(self.preview_widgets):
            return
        if widget_index >= len(self.composited_images):
            return

        diff_indices = self._get_diff_indices()
        if widget_index >= len(diff_indices):
            return

        diff_idx = diff_indices[widget_index]
        if diff_idx >= len(self.items):
            return

        composited = self.composited_images[widget_index]
        diff_image = self.items[diff_idx].aligned_image

        if self._onion_opacity <= 0:
            display = composited
        elif self._onion_opacity >= 1.0:
            display = diff_image
        else:
            display = self._blend_onion_skin(composited, diff_image)

        self.preview_widgets[widget_index].set_base_image(display)

    def _load_settings(self):
        """保存済み設定を読み込む"""
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
        self.check_auto_trim.setChecked(self.settings.value('save/auto_trim', True, type=bool))
        self.spin_trim_margin.setValue(self.settings.value('save/trim_margin', 0, type=int))
        self.spin_issue_threshold.setValue(self.settings.value('align/issue_threshold', 0.6, type=float))
        self.slider_onion.setValue(self.settings.value('ui/onion_opacity', 0, type=int))

        self._update_alignment_summary()

    def _save_settings(self):
        """設定を保存"""
        self.settings.setValue('window/geometry', self.saveGeometry())
        self.settings.setValue('window/splitter_sizes', self.main_splitter.sizes())

        self.settings.setValue('paths/input_dir', self._last_input_dir)
        self.settings.setValue('paths/output_dir', self._last_output_dir)

        self.settings.setValue('ui/brush_size', self.spin_brush_size.value())
        self.settings.setValue('ui/feather', self.slider_feather.value())
        self.settings.setValue('save/auto_trim', self.check_auto_trim.isChecked())
        self.settings.setValue('save/trim_margin', self.spin_trim_margin.value())
        self.settings.setValue('align/issue_threshold', self.spin_issue_threshold.value())
        self.settings.setValue('ui/onion_opacity', self.slider_onion.value())

    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName('Mask Composer')
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
        QTabWidget::pane { border: 1px solid #3e3e42; background-color: #1e1e1e; }
        QTabBar::tab { background-color: #2d2d30; color: #ccc; padding: 8px 16px; border: 1px solid #3e3e42; }
        QTabBar::tab:selected { background-color: #1e1e1e; border-bottom: 2px solid #007acc; }
        QRadioButton { color: #ccc; }
        QSpinBox { background-color: #3c3c3c; border: 1px solid #3e3e42; padding: 3px; }
        QSlider::groove:horizontal { background: #3c3c3c; height: 6px; border-radius: 3px; }
        QSlider::handle:horizontal { background: #0e639c; width: 14px; margin: -4px 0; border-radius: 7px; }
    """)

    window = MaskComposerWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
