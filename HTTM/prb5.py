# ==============================================================================
# Bài 5: Mô hình hỗn hợp Gaussian (GMM) với ước lượng tham số EM
#
# Mô tả:
# - Cài đặt thuật toán Expectation-Maximization (EM) cho mô hình GMM
# - Yêu cầu 1: Bước E (Expectation) tính xác suất hậu nghiệm gamma_ik
# - Yêu cầu 2: Bước M (Maximization) cập nhật pi_k, mu_k, Sigma_k (Weighted MLE)
# - Yêu cầu 3: Áp dụng lên dữ liệu 2D giả lập, trực quan hóa các đường đồng mức
#              (contour plots) qua các vòng lặp EM cho đến khi hội tụ
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

# Cố định random seed để kết quả mô phỏng có thể tái lập
np.random.seed(42)

# ==============================================================================
# 1. Hàm mật độ xác suất Gaussian nhiều chiều (Multivariate Gaussian PDF)
# ==============================================================================
def multivariate_gaussian_pdf(X, mean, cov):
    """
    Tính hàm mật độ xác suất N(x | mean, cov) cho mảng các điểm dữ liệu X.
    """
    d = len(mean)
    # Cộng thêm một lượng nhỏ epsilon vào đường chéo để ổn định số học
    cov_reg = cov + 1e-6 * np.eye(d)
    diff = X - mean
    inv_cov = np.linalg.inv(cov_reg)
    det_cov = np.linalg.det(cov_reg)
    
    norm_const = 1.0 / (np.power(2.0 * np.pi, d / 2.0) * np.sqrt(np.maximum(det_cov, 1e-12)))
    exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
    return norm_const * np.exp(np.clip(exponent, -50.0, 50.0))

# ==============================================================================
# YÊU CẦU 1 & 2: Cài đặt lớp GaussianMixtureModel với thuật toán EM
# ==============================================================================
class GaussianMixtureModel:
    """
    Mô hình hỗn hợp Gaussian (GMM) cài đặt từ đầu bằng NumPy.
    """
    def __init__(self, K=3, max_iters=50, tol=1e-4):
        self.K = K
        self.max_iters = max_iters
        self.tol = tol
        self.pi = None       # Trọng số cụm pi_k (shape: K)
        self.mu = None       # Vector kỳ vọng mu_k (shape: K, D)
        self.sigma = None    # Ma trận hiệp phương sai Sigma_k (shape: K, D, D)
        self.gamma = None    # Xác suất hậu nghiệm (shape: N, K)
        self.log_likelihood_history = []
        self.snapshots = []  # Lưu lại trạng thái để trực quan hóa

    def fit(self, X):
        N, D = X.shape
        
        # Khởi tạo tham số
        rand_indices = np.random.choice(N, self.K, replace=False)
        self.mu = X[rand_indices].copy()
        
        # Khởi tạo hiệp phương sai theo phương sai toàn cục
        global_var = np.var(X, axis=0)
        self.sigma = np.array([np.diag(global_var) for _ in range(self.K)])
        self.pi = np.ones(self.K) / self.K

        for it in range(self.max_iters):
            # ------------------------------------------------------------------
            # YÊU CẦU 1: BƯỚC E (Expectation Step)
            # Tính xác suất hậu nghiệm gamma_ik = P(z_i = k | x_i)
            # ------------------------------------------------------------------
            weighted_pdf = np.zeros((N, self.K))
            for k in range(self.K):
                weighted_pdf[:, k] = self.pi[k] * multivariate_gaussian_pdf(X, self.mu[k], self.sigma[k])
            
            total_density = np.sum(weighted_pdf, axis=1, keepdims=True)
            total_density = np.maximum(total_density, 1e-12)
            self.gamma = weighted_pdf / total_density  # shape (N, K)

            # Tính Log-Likelihood
            log_likelihood = np.sum(np.log(total_density))
            self.log_likelihood_history.append(log_likelihood)

            # Lưu lại trạng thái tại các vòng lặp tiêu biểu
            if it in [0, 1, 2, 4, 9, 14] or it == self.max_iters - 1:
                self.snapshots.append({
                    'iter': it + 1,
                    'mu': self.mu.copy(),
                    'sigma': self.sigma.copy(),
                    'pi': self.pi.copy(),
                    'gamma': self.gamma.copy(),
                    'll': log_likelihood
                })

            # Kiểm tra hội tụ
            if it > 0 and abs(self.log_likelihood_history[-1] - self.log_likelihood_history[-2]) < self.tol:
                print(f"Thuật toán EM đã hội tụ tại vòng lặp thứ {it + 1}!")
                break

            # ------------------------------------------------------------------
            # YÊU CẦU 2: BƯỚC M (Maximization Step - Weighted MLE)
            # Cập nhật lại pi_k, mu_k, Sigma_k
            # ------------------------------------------------------------------
            N_k = np.sum(self.gamma, axis=0)  # Tổng trách nhiệm cụm k

            # 1. Cập nhật trọng số cụm: pi_k = N_k / N
            self.pi = N_k / N

            # 2. Cập nhật vector kỳ vọng: mu_k = (1 / N_k) * sum_i (gamma_ik * x_i)
            for k in range(self.K):
                self.mu[k] = np.sum(self.gamma[:, k:k+1] * X, axis=0) / N_k[k]

            # 3. Cập nhật ma trận hiệp phương sai (Weighted Covariance):
            #    Sigma_k = (1 / N_k) * sum_i gamma_ik * (x_i - mu_k)(x_i - mu_k)^T
            for k in range(self.K):
                diff = X - self.mu[k]
                weighted_diff = self.gamma[:, k:k+1] * diff
                self.sigma[k] = (weighted_diff.T @ diff) / N_k[k]
                self.sigma[k] += 1e-6 * np.eye(D)  # Regularization

        return self

# ==============================================================================
# YÊU CẦU 3: Tạo tập dữ liệu 2D giả lập gồm 3 cụm Gaussian
# ==============================================================================
n_per_cluster = 150

# Cụm 1: Hướng nghiêng chéo
true_mu1 = np.array([1.0, 2.0])
true_cov1 = np.array([[1.5, 0.9], [0.9, 1.0]])

# Cụm 2: Dạng elip dẹt nằm ngang
true_mu2 = np.array([7.0, 1.5])
true_cov2 = np.array([[2.0, -0.6], [-0.6, 0.5]])

# Cụm 3: Cụm phía trên
true_mu3 = np.array([4.0, 7.0])
true_cov3 = np.array([[0.8, 0.2], [0.2, 1.2]])

X1 = np.random.multivariate_normal(true_mu1, true_cov1, n_per_cluster)
X2 = np.random.multivariate_normal(true_mu2, true_cov2, n_per_cluster)
X3 = np.random.multivariate_normal(true_mu3, true_cov3, n_per_cluster)
X = np.vstack([X1, X2, X3])

print("=" * 80)
print(f"Tổng số mẫu dữ liệu: {len(X)} điểm (mỗi cụm {n_per_cluster} điểm).")
print(f"Tọa độ tâm thực tế (Ground Truth Means):")
print(f"  Cụm 1: {true_mu1}")
print(f"  Cụm 2: {true_mu2}")
print(f"  Cụm 3: {true_mu3}")
print("=" * 80)

# ==============================================================================
# Huấn luyện mô hình GMM bằng thuật toán EM
# ==============================================================================
gmm = GaussianMixtureModel(K=3, max_iters=50, tol=1e-4)
gmm.fit(X)

print("\n" + "=" * 80)
print("KẾT QUẢ ƯỚC LƯỢNG THAM SỐ GMM SAU KHI HỘI TỤ:")
print("=" * 80)
for k in range(gmm.K):
    print(f"Cụm {k+1}:")
    print(f"  - Trọng số pi_{k+1}:  {gmm.pi[k]:.4f} (Thực tế: 0.3333)")
    print(f"  - Tâm kỳ vọng mu_{k+1}: [{gmm.mu[k, 0]:.3f}, {gmm.mu[k, 1]:.3f}]")
    print(f"  - Ma trận hiệp phương sai Sigma_{k+1}:")
    print(f"    [{gmm.sigma[k, 0, 0]:.3f}, {gmm.sigma[k, 0, 1]:.3f}]")
    print(f"    [{gmm.sigma[k, 1, 0]:.3f}, {gmm.sigma[k, 1, 1]:.3f}]")
print("=" * 80)

# ==============================================================================
# YÊU CẦU 3: Trực quan hóa các đường đồng mức (Contour Plots) qua các vòng lặp
# ==============================================================================
selected_snapshots = gmm.snapshots[:4]
n_plots = len(selected_snapshots)

fig, axes = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 5), dpi=120)

x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 120), np.linspace(y_min, y_max, 120))
grid_points = np.column_stack([xx.ravel(), yy.ravel()])
colors = ['#e74c3c', '#2ecc71', '#3498db']

for idx, snap in enumerate(selected_snapshots):
    ax = axes[idx]
    hard_labels = np.argmax(snap['gamma'], axis=1)
    
    for k in range(gmm.K):
        cluster_points = X[hard_labels == k]
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                   s=20, color=colors[k], alpha=0.55, edgecolors='none')
        
        pdf_grid = multivariate_gaussian_pdf(grid_points, snap['mu'][k], snap['sigma'][k])
        Z = pdf_grid.reshape(xx.shape)
        ax.contour(xx, yy, Z, levels=3, colors=colors[k], linewidths=2.2, alpha=0.9)
        ax.scatter(snap['mu'][k, 0], snap['mu'][k, 1], 
                   marker='X', s=120, color='black', edgecolors='white', linewidth=1.5, zorder=5)
        
    ax.set_title(f"Vòng lặp EM: {snap['iter']}\nLog-Likelihood: {snap['ll']:.1f}", 
                 fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel('$x_1$', fontsize=11)
    ax.set_ylabel('$x_2$', fontsize=11)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, linestyle=':', alpha=0.5)

plt.suptitle('Sự tiến hóa của các đường đồng mức Gaussian qua các vòng lặp EM', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

# Lưu đồ thị tiến hóa contour ra file ảnh
plt.savefig('/home/sum/project/ai-notebooks/HTTM/gmm_em_contours.png', dpi=150, bbox_inches='tight')
print("Đã lưu đồ thị đường đồng mức GMM vào: /home/sum/project/ai-notebooks/HTTM/gmm_em_contours.png")

# Đồ thị đường cong hội tụ Log-Likelihood
plt.figure(figsize=(8, 4.5), dpi=120)
plt.plot(range(1, len(gmm.log_likelihood_history) + 1), gmm.log_likelihood_history, 
         marker='o', color='purple', linewidth=2, markersize=5)
plt.title('Đường cong hội tụ của hàm Log-Likelihood theo vòng lặp EM', fontsize=13, fontweight='bold')
plt.xlabel('Số vòng lặp (Iterations)', fontsize=11)
plt.ylabel('Incomplete Log-Likelihood', fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()

plt.savefig('/home/sum/project/ai-notebooks/HTTM/gmm_log_likelihood.png', dpi=150)
print("Đã lưu đồ thị Log-Likelihood vào: /home/sum/project/ai-notebooks/HTTM/gmm_log_likelihood.png")
