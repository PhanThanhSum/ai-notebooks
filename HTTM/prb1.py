# ==============================================================================
# Bài 1: Trực quan hóa hiện tượng quá khớp (Overfitting) của MLE so với MAP
#
# 1. Cơ sở lý thuyết:
# - Tung đồng xu Bernoulli với xác suất ngửa thực tế theta_true = 0.7
# - k: số lần ngửa trong N lần tung (k = sum(x_i))
# - MLE: theta_mle = k / N
# - MAP (với tiên nghiệm Beta(alpha=2, beta=2)):
#   theta_map = (k + alpha - 1) / (N + alpha + beta - 2) = (k + 1) / (N + 2)
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

# Cố định random seed để kết quả mô phỏng có thể tái lập
np.random.seed(42)

# ==============================================================================
# YÊU CẦU 1: Tạo hàm mô phỏng tung đồng xu N lần
# ==============================================================================
def simulate_coin_toss(n, theta_true=0.7):
    """
    Mô phỏng n lần tung đồng xu độc lập.
    - Giá trị 1: Mặt ngửa (Heads) với xác suất theta_true
    - Giá trị 0: Mặt sấp (Tails) với xác suất 1 - theta_true
    """
    return np.random.binomial(n=1, p=theta_true, size=n)

# ==============================================================================
# YÊU CẦU 2: Tính toán theta_MLE và theta_MAP khi N tăng dần từ 1 đến 100
# ==============================================================================
theta_true = 0.7
max_N = 100
alpha = 2
beta = 2

# Mô phỏng một chuỗi 100 lần tung đồng xu liên tiếp
toss_results = simulate_coin_toss(max_N, theta_true)

# Tính tổng tích lũy số mặt ngửa (k) sau mỗi lần tung thứ N
k_cumulative = np.cumsum(toss_results)
N_values = np.arange(1, max_N + 1)

# 1. Ước lượng MLE: k / N
theta_mle = k_cumulative / N_values

# 2. Ước lượng MAP: (k + alpha - 1) / (N + alpha + beta - 2) = (k + 1) / (N + 2)
theta_map = (k_cumulative + alpha - 1) / (N_values + alpha + beta - 2)

print(f"Tổng số lần tung: {max_N}")
print(f"Số mặt ngửa quan sát được sau {max_N} lần: {k_cumulative[-1]}")
print(f"Ước lượng MLE tại N = {max_N}: {theta_mle[-1]:.4f}")
print(f"Ước lượng MAP tại N = {max_N}: {theta_map[-1]:.4f}")

# ==============================================================================
# YÊU CẦU 3: Sử dụng matplotlib để vẽ đồ thị so sánh ước lượng MLE và MAP theo N
# ==============================================================================
plt.figure(figsize=(12, 6), dpi=120)

# 1. Đường giá trị xác suất thực tế theta_true
plt.axhline(y=theta_true, color='crimson', linestyle='--', linewidth=2, 
            label=f'Giá trị thực tế ($\\theta_{{true}} = {theta_true}$)')

# 2. Đường ước lượng MLE
plt.plot(N_values, theta_mle, color='royalblue', marker='o', markersize=3.5, 
         linestyle='-', linewidth=1.8, label=r'MLE: $\hat{\theta}_{MLE} = \frac{k}{N}$')

# 3. Đường ước lượng MAP
plt.plot(N_values, theta_map, color='forestgreen', marker='s', markersize=3.5, 
         linestyle='-', linewidth=1.8, label=r'MAP: $\hat{\theta}_{MAP} = \frac{k+1}{N+2}$ (Beta(2, 2))')

# Cấu hình giao diện đồ thị
plt.title(r'So sánh ước lượng MLE và MAP theo số lượng mẫu $N$ ($\theta_{true} = 0.7$)', 
          fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Số lượng mẫu $N$ (Số lần tung đồng xu)', fontsize=12)
plt.ylabel(r'Giá trị ước lượng xác suất $\hat{\theta}$', fontsize=12)
plt.xlim(1, max_N)
plt.ylim(-0.05, 1.05)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11, loc='lower right', framealpha=0.9)
plt.tight_layout()

# Lưu đồ thị ra file ảnh để xem trực quan nếu cần
plt.savefig('/home/sum/project/ai-notebooks/HTTM/mle_vs_map.png', dpi=150)
print("Đã lưu đồ thị vào: /home/sum/project/ai-notebooks/HTTM/mle_vs_map.png")

# Hiển thị đồ thị
plt.show()
