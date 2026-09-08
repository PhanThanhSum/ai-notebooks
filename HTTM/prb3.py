# ==============================================================================
# Bài 3: Hồi quy Ridge Regression (MAP) và Linear Regression (MLE)
#
# Mô tả:
# - Giải bài toán hồi quy tuyến tính bằng phương pháp giải tích (closed-form)
# - Yêu cầu 1: Cài đặt hàm nghiệm OLS (Ordinary Least Squares - ứng với MLE)
# - Yêu cầu 2: Cài đặt hàm nghiệm Ridge Regression (ứng với MAP với Gaussian prior)
# - Yêu cầu 3: Tạo tập dữ liệu giả lập có nhiễu lớn và đa cộng tuyến, so sánh
#              trực quan trọng số w và sai số MSE giữa OLS và Ridge
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

# Cố định random seed để kết quả mô phỏng có thể tái lập
np.random.seed(42)

# ==============================================================================
# YÊU CẦU 1: Cài đặt hàm tính nghiệm OLS (MLE)
# ==============================================================================
def solve_ols(X, y):
    """
    Tính nghiệm OLS (Ordinary Least Squares - ứng với MLE).
    Công thức ma trận: w = (X^T * X)^(-1) * X^T * y
    """
    return np.linalg.inv(X.T @ X) @ X.T @ y

# ==============================================================================
# YÊU CẦU 2: Cài đặt hàm tính nghiệm Ridge Regression (MAP)
# ==============================================================================
def solve_ridge(X, y, lmbda):
    """
    Tính nghiệm Ridge Regression (ứng với MAP với Gaussian prior w ~ N(0, tau^2 * I)).
    Công thức ma trận: w = (X^T * X + lambda * I)^(-1) * X^T * y
    """
    n_features = X.shape[1]
    I = np.eye(n_features)
    return np.linalg.inv(X.T @ X + lmbda * I) @ X.T @ y

def compute_mse(y_true, y_pred):
    """Tính sai số toàn phương trung bình MSE."""
    return np.mean((y_true - y_pred) ** 2)

# ==============================================================================
# YÊU CẦU 3: Tạo tập dữ liệu giả lập có nhiễu lớn và đa cộng tuyến
# ==============================================================================
n_train = 60
n_test = 40
n_samples = n_train + n_test

# 1. Sinh các biến tiềm ẩn độc lập
z1 = np.random.randn(n_samples)
z2 = np.random.randn(n_samples)
z3 = np.random.randn(n_samples)

# 2. Tạo 5 đặc trưng có đa cộng tuyến cực cao:
# - x0 và x1 có tương quan gần như tuyệt đối (corr ~ 0.999)
x0 = z1 + 0.01 * np.random.randn(n_samples)
x1 = z1 + 0.01 * np.random.randn(n_samples)
# - x2 và x3 có tương quan mạnh
x2 = z2 + 0.02 * np.random.randn(n_samples)
x3 = 0.8 * z2 + 0.2 * z3 + 0.02 * np.random.randn(n_samples)
# - x4 độc lập
x4 = z3

X = np.column_stack([x0, x1, x2, x3, x4])

# 3. Trọng số thực tế (Ground Truth)
w_true = np.array([2.5, 2.5, -3.0, 1.5, 4.0])

# 4. Thêm nhiễu Gaussian lớn (sigma = 2.0)
noise_std = 2.0
noise = np.random.normal(0, noise_std, size=n_samples)
y = X @ w_true + noise

# 5. Chia tập Train / Test
X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# Đánh giá chỉ số điều kiện Condition Number
cond_number = np.linalg.cond(X_train.T @ X_train)
print("=" * 95)
print(f"Kích thước tập Train: {X_train.shape} | Tập Test: {X_test.shape}")
print(f"Chỉ số điều kiện Condition Number của X_train^T * X_train: {cond_number:.2f}")
print("(Condition Number > 1000 cho thấy hiện tượng đa cộng tuyến cực kỳ nghiêm trọng!)")
print("=" * 95)

# ==============================================================================
# YÊU CẦU 3: So sánh nghiệm OLS và Ridge với các mức lambda khác nhau
# ==============================================================================

# 1. Nghiệm OLS (MLE)
w_ols = solve_ols(X_train, y_train)
mse_ols_train = compute_mse(y_train, X_train @ w_ols)
mse_ols_test = compute_mse(y_test, X_test @ w_ols)

# 2. Khảo sát Ridge Regression trên dải lambda rộng
lambdas = np.logspace(-3, 3, 100)
weights_ridge = []
mse_train_list = []
mse_test_list = []

for lmb in lambdas:
    w_r = solve_ridge(X_train, y_train, lmb)
    weights_ridge.append(w_r)
    mse_train_list.append(compute_mse(y_train, X_train @ w_r))
    mse_test_list.append(compute_mse(y_test, X_test @ w_r))

weights_ridge = np.array(weights_ridge)

best_idx = np.argmin(mse_test_list)
best_lambda = lambdas[best_idx]
best_w_ridge = weights_ridge[best_idx]
best_mse_test = mse_test_list[best_idx]

# Bảng so sánh một số mức lambda tiêu biểu
representative_lambdas = [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
print(f"{'Mô hình / Lambda':<20} | {'Trọng số w ước lượng':<42} | {'MSE Train':<10} | {'MSE Test':<10}")
print("-" * 95)
print(f"{'Ground Truth (w_true)':<20} | {str(np.round(w_true, 2)):<42} | {'-':<10} | {'-':<10}")
print(f"{'OLS (MLE)':<20} | {str(np.round(w_ols, 2)):<42} | {mse_ols_train:<10.4f} | {mse_ols_test:<10.4f}")
print("-" * 95)

for lmb in representative_lambdas:
    w_r = solve_ridge(X_train, y_train, lmb)
    tr_mse = compute_mse(y_train, X_train @ w_r)
    te_mse = compute_mse(y_test, X_test @ w_r)
    highlight = " <-- [TỐT NHẤT]" if abs(lmb - best_lambda) < 1.0 else ""
    print(f"{'Ridge (lambda=' + str(lmb) + ')':<20} | {str(np.round(w_r, 2)):<42} | {tr_mse:<10.4f} | {te_mse:<10.4f}{highlight}")

print("=" * 95)
print(f"-> Lambda tối ưu nhất: lambda = {best_lambda:.3f} với MSE Test = {best_mse_test:.4f}\n")

# ==============================================================================
# YÊU CẦU 3: Trực quan hóa kết quả bằng Matplotlib
# ==============================================================================
plt.figure(figsize=(16, 11), dpi=120)

# ĐỒ THỊ 1: Sự biến thiên MSE theo Lambda (Bias-Variance Tradeoff)
plt.subplot(2, 2, 1)
plt.plot(lambdas, mse_train_list, label='Train MSE (Ridge)', color='blue', linewidth=2)
plt.plot(lambdas, mse_test_list, label='Test MSE (Ridge)', color='red', linewidth=2)
plt.axhline(y=mse_ols_test, color='darkred', linestyle='--', label=f'Test MSE (OLS = {mse_ols_test:.3f})')
plt.axvline(x=best_lambda, color='green', linestyle=':', linewidth=2, label=f'Best $\\lambda = {best_lambda:.2f}$')
plt.xscale('log')
plt.title(r'1. Sai số MSE theo hệ số điều chuẩn $\lambda$', fontsize=13, fontweight='bold')
plt.xlabel(r'$\lambda$ (Log scale)', fontsize=11)
plt.ylabel('Mean Squared Error (MSE)', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=9)

# ĐỒ THỊ 2: Đường vết trọng số (Ridge Regularization Path / Shrinkage)
plt.subplot(2, 2, 2)
for i in range(X.shape[1]):
    plt.plot(lambdas, weights_ridge[:, i], label=f'$w_{i}$ (True={w_true[i]})', linewidth=1.8)
plt.axvline(x=best_lambda, color='green', linestyle=':', linewidth=2, label=f'Best $\\lambda = {best_lambda:.2f}$')
plt.xscale('log')
plt.title(r'2. Vết trọng số (Weight Path) khi tăng $\lambda$', fontsize=13, fontweight='bold')
plt.xlabel(r'$\lambda$ (Log scale)', fontsize=11)
plt.ylabel(r'Giá trị trọng số $w_i$', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=9, loc='upper right')

# ĐỒ THỊ 3: So sánh trực quan giá trị trọng số w giữa w_true, OLS và Ridge
plt.subplot(2, 1, 2)
features = [f'x_{i}' for i in range(X.shape[1])]
x_pos = np.arange(len(features))
width = 0.25

plt.bar(x_pos - width, w_true, width, label=r'Ground Truth ($w_{true}$)', color='#2ecc71', alpha=0.9)
plt.bar(x_pos, w_ols, width, label=r'OLS (MLE - $\lambda = 0$)', color='#e74c3c', alpha=0.85)
plt.bar(x_pos + width, best_w_ridge, width, label=f'Ridge (MAP - $\\lambda = {best_lambda:.2f}$)', color='#3498db', alpha=0.9)

plt.title(r'3. So sánh trực quan độ lớn trọng số $w$ giữa Ground Truth, OLS và Ridge', fontsize=13, fontweight='bold')
plt.xlabel('Các đặc trưng (Features)', fontsize=11)
plt.ylabel('Giá trị trọng số', fontsize=11)
plt.xticks(x_pos, features, fontsize=11)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.legend(fontsize=11)

plt.tight_layout()

# Lưu đồ thị ra file ảnh
plt.savefig('/home/sum/project/ai-notebooks/HTTM/ridge_vs_ols.png', dpi=150)
print("Đã lưu đồ thị trực quan hóa vào: /home/sum/project/ai-notebooks/HTTM/ridge_vs_ols.png")
