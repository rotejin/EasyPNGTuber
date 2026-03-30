"""
位置合わせプロセッサー
AKAZE/ORB特徴点マッチングで画像の位置ずれを補正
"""
import cv2
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class AlignConfig:
    """位置合わせ設定"""
    # AKAZEパラメータ
    akaze_threshold: float = 0.001
    akaze_n_octaves: int = 4
    akaze_n_octave_layers: int = 4
    
    # ORBパラメータ（フォールバック用）
    orb_n_features: int = 5000
    orb_scale_factor: float = 1.2
    orb_n_levels: int = 8
    
    # マッチングパラメータ
    knn_ratio: float = 0.75
    
    # RANSACパラメータ
    ransac_threshold: float = 3.0
    ransac_max_iters: int = 2000
    ransac_confidence: float = 0.99
    min_inliers: int = 10
    
    # 成功判定
    success_score_threshold: float = 0.6
    warning_score_threshold: float = 0.3
    
    # 変換制限
    max_rotation_deg: float = 30.0
    max_scale: float = 1.2
    min_scale: float = 0.8


class Aligner:
    """画像位置合わせクラス"""
    
    def __init__(self, config: Optional[AlignConfig] = None):
        self.config = config or AlignConfig()
    
    def align(self, base_image: np.ndarray, target_image: np.ndarray,
              use_orb: bool = False, base_mask: Optional[np.ndarray] = None) -> dict:
        """
        位置合わせを実行

        Args:
            base_image: ベース画像（BGRまたはグレースケール）
            target_image: 位置合わせ対象画像
            use_orb: Trueの場合ORBを使用、Falseの場合AKAZE
            base_mask: ベース画像用ROIマスク（白=255の領域で特徴点検出）
                       ※ターゲット画像は全体から検出（座標系が異なるため）

        Returns:
            結果辞書
        """
        # グレースケール化
        base_gray = self._to_grayscale(base_image)
        target_gray = self._to_grayscale(target_image)

        if use_orb:
            return self._align_orb(base_gray, target_gray, base_mask)
        else:
            result = self._align_akaze(base_gray, target_gray, base_mask)
            # AKAZE失敗時はORBで再試行
            if not result['success'] and result.get('retry_recommended', False):
                return self._align_orb(base_gray, target_gray, base_mask)
            return result
    
    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """グレースケール変換"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def _align_akaze(self, base_gray: np.ndarray,
                     target_gray: np.ndarray,
                     base_mask: Optional[np.ndarray] = None) -> dict:
        """AKAZEで位置合わせ"""
        try:
            # AKAZE検出器
            akaze = cv2.AKAZE_create(
                descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
                descriptor_size=0,
                descriptor_channels=3,
                threshold=self.config.akaze_threshold,
                nOctaves=self.config.akaze_n_octaves,
                nOctaveLayers=self.config.akaze_n_octave_layers,
                diffusivity=cv2.DIFF_PM_G2
            )

            # 特徴点検出（baseのみマスク適用、targetは全体から検出）
            kp1, des1 = akaze.detectAndCompute(base_gray, base_mask)
            kp2, des2 = akaze.detectAndCompute(target_gray, None)
            
            if des1 is None or des2 is None:
                return self._error_result("特徴点検出失敗")
            
            if len(kp1) < self.config.min_inliers or len(kp2) < self.config.min_inliers:
                return self._error_result("特徴点が不足")
            
            # マッチング（KNN + 比率テスト）
            # AKAZE(MLDB)はバイナリ記述子のためHamming距離を使用
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            matches = matcher.knnMatch(des1, des2, k=2)
            
            # 良いマッチを選択（Lowe's ratio test）
            good_matches = []
            for m_n in matches:
                if len(m_n) == 2:
                    m, n = m_n
                    if m.distance < self.config.knn_ratio * n.distance:
                        good_matches.append(m)
            
            if len(good_matches) < self.config.min_inliers:
                return self._error_result("良いマッチが不足", retry_recommended=True)
            
            # 変換行列推定
            return self._estimate_transform(kp1, kp2, good_matches, "AKAZE")
            
        except Exception as e:
            return self._error_result(f"AKAZEエラー: {str(e)}", retry_recommended=True)
    
    def _align_orb(self, base_gray: np.ndarray,
                   target_gray: np.ndarray,
                   base_mask: Optional[np.ndarray] = None) -> dict:
        """ORBで位置合わせ（フォールバック）"""
        try:
            # ORB検出器
            orb = cv2.ORB_create(
                nfeatures=self.config.orb_n_features,
                scaleFactor=self.config.orb_scale_factor,
                nlevels=self.config.orb_n_levels,
                edgeThreshold=31,
                patchSize=31
            )

            # 特徴点検出（baseのみマスク適用、targetは全体から検出）
            kp1, des1 = orb.detectAndCompute(base_gray, base_mask)
            kp2, des2 = orb.detectAndCompute(target_gray, None)
            
            if des1 is None or des2 is None:
                return self._error_result("ORB特徴点検出失敗")
            
            if len(kp1) < self.config.min_inliers or len(kp2) < self.config.min_inliers:
                return self._error_result("ORB特徴点が不足")
            
            # マッチング（相互チェック）
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            if len(matches) < self.config.min_inliers:
                return self._error_result("ORBマッチが不足")
            
            # 上位100マッチを使用
            good_matches = matches[:min(100, len(matches))]
            
            # 変換行列推定
            return self._estimate_transform(kp1, kp2, good_matches, "ORB")
            
        except Exception as e:
            return self._error_result(f"ORBエラー: {str(e)}")
    
    def _estimate_transform(self, kp1, kp2, matches, method: str) -> dict:
        """変換行列を推定"""
        # 対応点を抽出
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # アフィン変換推定（RANSAC）
        matrix, mask = cv2.estimateAffinePartial2D(
            dst_pts, src_pts,  # target -> base
            method=cv2.RANSAC,
            ransacReprojThreshold=self.config.ransac_threshold,
            maxIters=self.config.ransac_max_iters,
            confidence=self.config.ransac_confidence
        )
        
        if matrix is None:
            return self._error_result("変換行列推定失敗")
        
        # インライアー数
        inliers = np.sum(mask) if mask is not None else 0
        total_matches = len(matches)
        
        # 変換パラメータ分解
        transform_info = self._decompose_matrix(matrix)
        
        # 制限チェック
        if not self._check_transform_limits(transform_info):
            return self._error_result("変換パラメータが制限を超過")
        
        # スコア計算
        score = self._calculate_score(inliers, total_matches, matrix, src_pts, dst_pts, mask)
        
        # 成功判定
        success = score >= self.config.success_score_threshold
        warning = self.config.warning_score_threshold <= score < self.config.success_score_threshold
        
        return {
            'success': success,
            'warning': warning and not success,
            'score': score,
            'method': method,
            'inliers': int(inliers),
            'total_matches': total_matches,
            'matrix': matrix,
            'transform': transform_info,
            'error_message': ''
        }
    
    def _decompose_matrix(self, matrix: np.ndarray) -> dict:
        """変換行列を分解"""
        # 簡易分解（厳密な分解は複雑なので近似）
        a, b, tx = matrix[0]
        c, d, ty = matrix[1]
        
        # スケール（近似）
        scale_x = np.sqrt(a**2 + b**2)
        scale_y = np.sqrt(c**2 + d**2)
        scale = (scale_x + scale_y) / 2
        
        # 回転（近似）
        rotation = np.arctan2(b, a) * 180 / np.pi
        
        return {
            'translation': [float(tx), float(ty)],
            'rotation_deg': float(rotation),
            'scale': float(scale)
        }
    
    def _check_transform_limits(self, transform: dict) -> bool:
        """変換パラメータが制限内かチェック"""
        rotation = abs(transform['rotation_deg'])
        scale = transform['scale']
        
        if rotation > self.config.max_rotation_deg:
            return False
        if scale < self.config.min_scale or scale > self.config.max_scale:
            return False
        return True
    
    def _calculate_score(self, inliers: int, total: int, matrix: np.ndarray,
                        src_pts, dst_pts, mask) -> float:
        """整合性スコアを計算"""
        if total == 0:
            return 0.0
        
        # 基本スコア：インライアー率
        base_score = inliers / total
        
        # 再投影誤差で補正
        if mask is not None and inliers > 0:
            inlier_src = src_pts[mask.ravel() == 1]
            inlier_dst = dst_pts[mask.ravel() == 1]
            
            # 変換適用
            transformed = cv2.transform(inlier_dst, matrix)
            errors = np.linalg.norm(inlier_src - transformed, axis=2)
            median_error = np.median(errors)
            
            # 誤差が小さいほどスコア向上（0-1pxで1.0、10pxで0.5）
            error_factor = max(0.5, 1.0 - median_error / 20.0)
        else:
            error_factor = 0.5
        
        score = base_score * error_factor
        return min(1.0, max(0.0, score))
    
    def _error_result(self, message: str, retry_recommended: bool = False) -> dict:
        """エラー結果を生成"""
        return {
            'success': False,
            'warning': False,
            'score': 0.0,
            'method': '',
            'inliers': 0,
            'total_matches': 0,
            'matrix': None,
            'transform': None,
            'error_message': message,
            'retry_recommended': retry_recommended
        }
    
    def apply_transform(self, image: np.ndarray, matrix: np.ndarray,
                       output_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """変換を適用"""
        if output_size is None:
            output_size = (image.shape[1], image.shape[0])
        
        # アルファチャンネル対応
        if len(image.shape) == 3 and image.shape[2] == 4:
            # RGBA
            result = cv2.warpAffine(
                image, matrix, output_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )
        else:
            result = cv2.warpAffine(
                image, matrix, output_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )
        
        return result

    def apply_transform_with_mask(
        self,
        image: np.ndarray,
        matrix: np.ndarray,
        output_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """変換結果と有効領域マスクを返す

        有効領域マスクは、変換後に「元画像由来の画素が存在する領域」を255で表す。
        """
        if output_size is None:
            output_size = (image.shape[1], image.shape[0])

        transformed = self.apply_transform(image, matrix, output_size)

        source_h, source_w = image.shape[:2]
        source_valid = np.full((source_h, source_w), 255, dtype=np.uint8)
        valid_mask = cv2.warpAffine(
            source_valid, matrix, output_size,
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

        return transformed, valid_mask


def _align_single_slice(args: tuple) -> dict:
    """Process a single slice alignment (for multiprocessing).

    Args:
        args: (index, base_bgr, target_bgr, target_bgra, base_size)

    Returns:
        dict with index, aligned_image, valid_mask, success, score
    """
    index, base_bgr, target_bgr, target_bgra, base_size = args
    aligner = Aligner(AlignConfig())

    result = aligner.align(base_bgr, target_bgr)

    if result['matrix'] is not None:
        aligned, valid_mask = aligner.apply_transform_with_mask(
            target_bgra, result['matrix'], base_size
        )
        return {
            'index': index,
            'aligned_image': aligned,
            'valid_mask': valid_mask,
            'success': bool(result['success']),
            'score': result['score'],
        }
    else:
        return {
            'index': index,
            'aligned_image': target_bgra.copy(),
            'valid_mask': np.full(target_bgra.shape[:2], 255, dtype=np.uint8),
            'success': False,
            'score': result.get('score', 0.0),
        }
