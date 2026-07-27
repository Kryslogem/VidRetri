#!/usr/bin/env python3
from modular_frame_extractor import ModularFrameExtractor, PassConstants

# Cấu hình BẢO TOÀN THÔNG TIN TỐI ĐA (High Information Preservation)
constants = PassConstants(
    pass1_target_fps=5.0,               # Pass 1: Lấy mẫu dày 7 FPS
    pass2_ecr_threshold=0.14,           # Pass 2: Độ nhạy cao bắt sự thay đổi nhỏ
    pass2_suppress_camera_motion=False, # 
    pass3_min_sharpness=50.0,           # Pass 3: Xoá frame quá mờ
    pass4_cosine_distance=0.04,         # Gộp frame có ít khác biệt bằng CLIP
    pass5_similarity_threshold=0.97     # So sánh SSIM + Màu sắc + dHash
)

# Khởi tạo extractor
extractor = ModularFrameExtractor(constants=constants)

# Chạy trích xuất
keyframes = extractor.run(
    video_path=r"D:\.HandOnRAG\frameExtract\video\VID001.mp4",
    output_dir="frames"
)