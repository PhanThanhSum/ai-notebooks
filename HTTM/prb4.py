# ==============================================================================
# Bài 4: Tối ưu hóa Hồi quy Logistic với L2 Regularization (MAP) bằng Gradient Ascent
#
# Mô tả:
# - Cài đặt thuật toán Logistic Regression cho phân loại nhị phân
# - Yêu cầu 1: Xây dựng hàm tính Log-posterior (Bernoulli Likelihood + L2 Prior)
# - Yêu cầu 2: Viết thuật toán Gradient Ascent để cực đại hóa Log-posterior theo w
# - Yêu cầu 3: Thử nghiệm trên tập dữ liệu phi tuyến tính make_moons với mở rộng đa thức,
#              vẽ đường biên quyết định và minh họa cách lambda kiểm soát overfitting
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

# Cố định random seed để kết quả mô phỏng có thể tái lập
np.random.seed(42)

# ==============================================================================
# YÊU CẦU 1: Hàm kích hoạt Sigmoid & Hàm tính Log-posterior
# ==============================================================================
def sigmoid(z):
    """Hàm kích hoạt Sigmoid an toàn chống tràn số (overflow)."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -25.0, 25.0)))

def compute_log_posterior(w, X, y, lmbda):
    """
    YÊU CẦU 1: Tính hàm mục tiêu Log-posterior:
    J(w) = Log-likelihood (Bernoulli) - (lambda / 2) * ||w_reg||^2
    (Không phạt trọng số bias w[0])
    """
    p = sigmoid(X @ w)
    eps = 1e-15  # Tránh lỗi log(0)
    p = np.clip(p, eps, 1.0 - eps)
    
    # 1. Thành phần Log-likelihood Bernoulli: sum [ y*ln(p) + (1-y)*ln(1-p) ]
    log_likelihood = np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    
    # 2. Thành phần tiên nghiệm chuẩn Gaussian Prior (L2 regularization)
    log_prior = -0.5 * lmbda * np.sum(w[1:] ** 2)
    
    return log_likelihood + log_prior

# ==============================================================================
# YÊU CẦU 2: Thuật toán Gradient Ascent
# ==============================================================================
def train_logistic_regression_map(X, y, lmbda=1.0, lr=0.8, n_iters=3000):
    """
    YÊU CẦU 2: Thuật toán Gradient Ascent tối ưu hóa hàm mục tiêu theo trọng số w.
    Quy tắc cập nhật: w = w + lr * (grad / N)
    """
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    history_log_posterior = []

    for it in range(n_iters):
        p = sigmoid(X @ w)
        
        # 1. Gradient của Log-likelihood: X^T (y - p)
        grad_likelihood = X.T @ (y - p)
        
        # 2. Gradient của Log-prior L2: - lambda * w_reg
        grad_prior = np.zeros_like(w)
        grad_prior[1:] = -lmbda * w[1:]
        
        # 3. Gradient tổng của Log-posterior
        grad = grad_likelihood + grad_prior
        
        # 4. Cập nhật Gradient Ascent (leo đồi để cực đại hóa J(w))
        w += lr * (grad / n_samples)
        
        # Ghi nhận giá trị Log-posterior để theo dõi hội tụ
        if it % 50 == 0 or it == n_iters - 1:
            history_log_posterior.append(compute_log_posterior(w, X, y, lmbda))
            
    return w, history_log_posterior

def predict(X, w):
    """Dự đoán nhãn nhị phân dựa trên ngưỡng xác suất 0.5."""
    return (sigmoid(X @ w) >= 0.5).astype(int)

# ==============================================================================
# YÊU CẦU 3: Dữ liệu make_moons & Mở rộng đặc trưng đa thức (Polynomial Features)
# ==============================================================================
def generate_make_moons(n_samples=200, noise=0.25, random_state=42):
    """Sinh tập dữ liệu hai nửa vầng trăng lồng vào nhau (make_moons)."""
    try:
        from sklearn.datasets import make_moons
        return make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    except ImportError:
        np.random.seed(random_state)
        n_out = n_samples // 2
        n_in = n_samples - n_out
        theta_out = np.linspace(0, np.pi, n_out)
        theta_in = np.linspace(0, np.pi, n_in)
        x_out = np.column_stack([np.cos(theta_out), np.sin(theta_out)])
        x_in = np.column_stack([1.0 - np.cos(theta_in), 1.0 - np.sin(theta_in) - 0.5])
        X = np.vstack([x_out, x_in])
        y = np.hstack([np.zeros(n_out, dtype=int), np.ones(n_in, dtype=int)])
        if noise:
            X += np.random.normal(scale=noise, size=X.shape)
        perm = np.random.permutation(n_samples)
        return X[perm], y[perm]

X_raw, y = generate_make_moons(n_samples=200, noise=0.25, random_state=42)

# Phân chia tập Train (70%) và Test (30%)
n_train = int(0.7 * len(X_raw))
X_train_raw, X_test_raw = X_raw[:n_train], X_raw[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# Hàm mở rộng đặc trưng đa thức bậc cao (Polynomial Feature Mapping)
def map_polynomial_features(X, degree=6):
    """
    Mở rộng không gian đặc trưng từ 2 chiều (x1, x2) sang đa thức bậc 6 (28 chiều):
    Phi(x) = [1, x1, x2, x1^2, x1*x2, x2^2, ..., x2^6]
    """
    x1 = X[:, 0]
    x2 = X[:, 1]
    out = [np.ones(len(x1))]  # Cột bias x0 = 1
    for d in range(1, degree + 1):
        for i in range(d + 1):
            out.append((x1 ** (d - i)) * (x2 ** i))
    return np.column_stack(out)

degree = 6
X_train = map_polynomial_features(X_train_raw, degree=degree)
X_test = map_polynomial_features(X_test_raw, degree=degree)

print("=" * 85)
print(f"Số lượng mẫu Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Số chiều đặc trưng sau khi mở rộng đa thức bậc {degree}: {X_train.shape[1]} chiều")
print("=" * 85)

# ==============================================================================
# YÊU CẦU 3: Thử nghiệm với các mức lambda khác nhau
# ==============================================================================
lambda_configs = [0.0, 1.0, 100.0]
results = {}

print(f"{'Lambda':<10} | {'Train Acc (%)':<15} | {'Test Acc (%)':<15} | {'Chuẩn ||w||_2':<15} | {'Nhận định':<20}")
print("-" * 85)

for lmb in lambda_configs:
    w_fit, history = train_logistic_regression_map(X_train, y_train, lmbda=lmb, lr=0.8, n_iters=4000)
    train_acc = np.mean(predict(X_train, w_fit) == y_train) * 100.0
    test_acc = np.mean(predict(X_test, w_fit) == y_test) * 100.0
    norm_w = np.linalg.norm(w_fit[1:])
    
    if lmb == 0.0:
        verdict = "Overfitting (Quá khớp)"
    elif lmb == 1.0:
        verdict = "Optimal (Tối ưu)"
    else:
        verdict = "Underfitting (Thiếu khớp)"
        
    print(f"{lmb:<10.1f} | {train_acc:<15.2f} | {test_acc:<15.2f} | {norm_w:<15.3f} | {verdict:<20}")
    results[lmb] = {'w': w_fit, 'history': history, 'train_acc': train_acc, 'test_acc': test_acc}

print("=" * 85)

# ==============================================================================
# YÊU CẦU 3: Vẽ đường biên quyết định (Decision Boundary) cho các mức lambda
# ==============================================================================
plt.figure(figsize=(18, 5.5), dpi=120)

# Tạo lưới tọa độ 2D
x1_min, x1_max = X_raw[:, 0].min() - 0.5, X_raw[:, 0].max() + 0.5
x2_min, x2_max = X_raw[:, 1].min() - 0.5, X_raw[:, 1].max() + 0.5
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 300), np.linspace(x2_min, x2_max, 300))
grid_points = np.column_stack([xx1.ravel(), xx2.ravel()])
grid_poly = map_polynomial_features(grid_points, degree=degree)

titles = [
    r"Overfitting ($\lambda = 0.0$ - MLE)" + "\nĐường biên phức tạp, bẫy nhiễu",
    r"Optimal Regularization ($\lambda = 1.0$ - MAP)" + "\nĐường biên mượt mà, phân tách tự nhiên",
    r"Underfitting ($\lambda = 100.0$ - MAP)" + "\nLực phạt quá lớn, đường biên bị co phẳng"
]

for idx, lmb in enumerate(lambda_configs):
    plt.subplot(1, 3, idx + 1)
    w_fit = results[lmb]['w']
    
    # Tính xác suất dự đoán trên lưới 2D
    probs = sigmoid(grid_poly @ w_fit).reshape(xx1.shape)
    
    # Nền xác suất
    plt.contourf(xx1, xx2, probs, levels=20, cmap='RdBu_r', alpha=0.3)
    # Đường biên quyết định P = 0.5
    plt.contour(xx1, xx2, probs, levels=[0.5], colors='black', linewidths=2.5)
    
    # Điểm dữ liệu
    plt.scatter(X_train_raw[y_train == 0, 0], X_train_raw[y_train == 0, 1], 
                color='blue', marker='o', edgecolors='k', s=45, label='Lớp 0 (Train)')
    plt.scatter(X_train_raw[y_train == 1, 0], X_train_raw[y_train == 1, 1], 
                color='red', marker='s', edgecolors='k', s=45, label='Lớp 1 (Train)')
    plt.scatter(X_test_raw[y_test == 0, 0], X_test_raw[y_test == 0, 1], 
                color='deepskyblue', marker='o', edgecolors='yellow', s=55, alpha=0.9, label='Lớp 0 (Test)')
    plt.scatter(X_test_raw[y_test == 1, 0], X_test_raw[y_test == 1, 1], 
                color='darkorange', marker='s', edgecolors='yellow', s=55, alpha=0.9, label='Lớp 1 (Test)')
    
    acc_info = f"Train Acc: {results[lmb]['train_acc']:.1f}% | Test Acc: {results[lmb]['test_acc']:.1f}%"
    plt.title(f"{titles[idx]}\n{acc_info}", fontsize=11, fontweight='bold', pad=10)
    plt.xlabel('$x_1$', fontsize=11)
    plt.ylabel('$x_2$', fontsize=11)
    if idx == 0:
        plt.legend(fontsize=8, loc='upper right', framealpha=0.85)
    plt.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()

# Lưu đồ thị ra file ảnh
plt.savefig('/home/sum/project/ai-notebooks/HTTM/logistic_map_boundaries.png', dpi=150)
print("Đã lưu đồ thị đường biên quyết định vào: /home/sum/project/ai-notebooks/HTTM/logistic_map_boundaries.png")
