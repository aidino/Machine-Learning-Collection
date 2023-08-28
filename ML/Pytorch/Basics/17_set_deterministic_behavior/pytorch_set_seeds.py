import random
import torch
import os
import numpy as np


def seed_everything(seed=42):
    # Dòng này đặt giá trị seed cho việc tính toán băm trong Python. 
    # Điều này ảnh hưởng đến cách các tập dữ liệu được băm và sắp xếp.
    os.environ["PYTHONHASHSEED"] = str(seed) 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Dòng này đặt giá trị seed cho việc tạo ngẫu nhiên trên GPU nếu có sử dụng PyTorch trên GPU.
    torch.cuda.manual_seed(seed)
    # Đây là việc đặt giá trị seed cho tất cả thiết bị GPU mà PyTorch hỗ trợ, 
    # giúp đồng nhất quá trình tạo ngẫu nhiên trên các thiết bị khác nhau.
    torch.cuda.manual_seed_all(seed)
    # Dòng này thiết lập chế độ xác định (deterministic) cho thư viện cuDNN của NVIDIA khi chạy trên GPU. 
    # Điều này đảm bảo rằng các phép toán cuDNN sẽ cho kết quả giống nhau trên các lần chạy khác nhau.
    torch.backends.cudnn.deterministic = True
    # Dòng này tắt tính năng tự động tối ưu hóa của cuDNN, 
    # để đảm bảo tính nhất quán và độ tin cậy cao hơn trong việc so sánh kết quả giữa các lần chạy.
    torch.backends.cudnn.benchmark = False


seed_everything()

# Do training etc after running seed_everything
