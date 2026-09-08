# ==============================================================================
# Bài 2: Phân loại văn bản Naive Bayes với Laplace Smoothing (MAP)
#
# Mô tả:
# - Xây dựng bộ phân loại Email Spam/Ham from scratch bằng NumPy
# - Yêu cầu 1: Tính xác suất tiên nghiệm P(C_k) bằng MLE và P(w_i | C_k)
# - Yêu cầu 2: Cơ chế Laplace Smoothing (MAP với prior Dirichlet đồng nhất)
# - Yêu cầu 3: Dự đoán (predict) và đánh giá độ chính xác trên tập test giả định
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import re

# ------------------------------------------------------------------------------
# 1. Hàm tiền xử lý văn bản (Tokenization)
# ------------------------------------------------------------------------------
def tokenize(text):
    """Chuyển thành chữ thường và trích xuất các từ vựng."""
    return re.findall(r'\b\w+\b', text.lower())

# ------------------------------------------------------------------------------
# 2. Xây dựng lớp Naive Bayes Classifier từ đầu (from scratch)
# ------------------------------------------------------------------------------
class NaiveBayesClassifier:
    """
    Mô hình Multinomial Naive Bayes xây dựng từ đầu bằng NumPy:
    - alpha = 1.0: Laplace Smoothing (MAP với Dirichlet prior đồng nhất)
    - alpha = 0.0: MLE thuần túy (không làm trơn)
    """
    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)
        self.classes = None
        self.class_priors = {}            # P(C_k)
        self.word_counts = {}            # count(w_i, C_k)
        self.total_words_in_class = {}   # sum_w count(w, C_k)
        self.vocab = set()               # Tập từ vựng V
        self.vocab_size = 0              # |V|
        self.word_probs = {}             # P(w_i | C_k)

    def fit(self, X, y):
        """
        YÊU CẦU 1: Tính xác suất tiên nghiệm P(C_k) bằng MLE và P(w_i | C_k).
        YÊU CẦU 2: Áp dụng cơ chế Laplace Smoothing (MAP).
        """
        self.classes, counts = np.unique(y, return_counts=True)
        total_docs = len(y)

        # 1. Tính xác suất tiên nghiệm của từng lớp P(C_k) bằng MLE
        #    P(C_k) = N_{C_k} / N
        for c, count in zip(self.classes, counts):
            self.class_priors[c] = count / total_docs
            self.word_counts[c] = {}
            self.total_words_in_class[c] = 0

        # 2. Xây dựng tập từ vựng V và đếm tần suất xuất hiện trong từng lớp
        for text, label in zip(X, y):
            words = tokenize(text)
            for w in words:
                self.vocab.add(w)
                self.word_counts[label][w] = self.word_counts[label].get(w, 0) + 1
                self.total_words_in_class[label] += 1

        self.vocab_size = len(self.vocab)

        # 3. Tính xác suất có điều kiện P(w_i | C_k) với Laplace Smoothing:
        #    P(w_i | C_k) = (count(w_i, C_k) + alpha) / (total_words_in_C_k + alpha * |V|)
        for c in self.classes:
            self.word_probs[c] = {}
            denominator = self.total_words_in_class[c] + self.alpha * self.vocab_size
            for w in self.vocab:
                count_w = self.word_counts[c].get(w, 0)
                if self.alpha == 0:
                    prob = count_w / denominator if denominator > 0 else 0.0
                else:
                    prob = (count_w + self.alpha) / denominator
                self.word_probs[c][w] = prob

    def predict_log_proba(self, text):
        """
        Tính log xác suất hậu nghiệm ln P(C_k | D) trên Log-space để tránh tràn số dưới:
        ln P(C_k | D) = ln P(C_k) + sum_i ln P(w_i | C_k)
        """
        words = tokenize(text)
        log_posteriors = {}

        for c in self.classes:
            # Bắt đầu với log xác suất tiên nghiệm
            log_score = np.log(self.class_priors[c])
            denominator = self.total_words_in_class[c] + self.alpha * self.vocab_size

            for w in words:
                if w in self.vocab:
                    prob = self.word_probs[c][w]
                else:
                    # Xử lý từ mới xuất hiện ngoài từ điển huấn luyện (OOV)
                    # Với Laplace Smoothing, từ mới được gán xác suất làm trơn tối thiểu
                    if self.alpha > 0 and denominator > 0:
                        prob = self.alpha / denominator
                    else:
                        prob = 0.0

                if prob > 0:
                    log_score += np.log(prob)
                else:
                    log_score += -np.inf

            log_posteriors[c] = log_score
        return log_posteriors

    def predict(self, X):
        """
        YÊU CẦU 3: Dự đoán nhãn có log-posterior cao nhất.
        """
        predictions = []
        for text in X:
            scores = self.predict_log_proba(text)
            best_class = max(scores, key=scores.get)
            predictions.append(best_class)
        return np.array(predictions)

    def evaluate(self, X, y):
        """Đánh giá độ chính xác (Accuracy)."""
        preds = self.predict(X)
        accuracy = np.mean(preds == np.array(y))
        return accuracy, preds

# ------------------------------------------------------------------------------
# 3. Chuẩn bị tập dữ liệu giả định (Toy Dataset)
# ------------------------------------------------------------------------------
train_X = [
    "buy cheap watches free shipping today",              # Spam
    "exclusive deal win cash prize claim now",             # Spam
    "congratulations winner claim free gift reward now",   # Spam
    "click here for free credit card loan deal",          # Spam
    "project meeting scheduled tomorrow morning",         # Ham
    "please send the quarterly report for review",       # Ham
    "team lunch together at office cafeteria",           # Ham
    "discussion on machine learning assignment project"   # Ham
]
train_y = np.array(["spam", "spam", "spam", "spam", "ham", "ham", "ham", "ham"])

# Tập test chứa cả từ đã học và từ mới (out-of-vocabulary)
test_X = [
    "claim your free cash prize today",           # Spam chuẩn (chứa free, cash, prize)
    "exclusive discount deal buy now",           # Spam có từ mới 'discount'
    "meeting tomorrow for project review",        # Ham chuẩn (chứa meeting, project, review)
    "can you send report before lunch"           # Ham có từ mới 'before', 'can'
]
test_y = np.array(["spam", "spam", "ham", "ham"])

# ------------------------------------------------------------------------------
# 4. Huấn luyện và Đánh giá mô hình MAP (Laplace Smoothing alpha = 1.0)
# ------------------------------------------------------------------------------
print("=" * 80)
print("HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH NAIVE BAYES VỚI LAPLACE SMOOTHING (MAP)")
print("=" * 80)

nb_model = NaiveBayesClassifier(alpha=1.0)
nb_model.fit(train_X, train_y)

print(f"Kích thước từ điển V: {nb_model.vocab_size} từ")
print(f"Xác suất tiên nghiệm P(Spam): {nb_model.class_priors['spam']:.2f}")
print(f"Xác suất tiên nghiệm P(Ham):  {nb_model.class_priors['ham']:.2f}\n")

# Đánh giá trên tập test
acc, preds = nb_model.evaluate(test_X, test_y)
print(f"Độ chính xác trên tập kiểm tra (Accuracy): {acc * 100:.2f}%\n")
print("Chi tiết từng mẫu kiểm tra:")
print("-" * 80)
for i, (text, true_lbl, pred_lbl) in enumerate(zip(test_X, test_y, preds)):
    scores = nb_model.predict_log_proba(text)
    status = "✓ ĐÚNG" if pred_lbl == true_lbl else "✗ SAI"
    print(f"Mẫu {i+1}: \"{text}\"")
    print(f"  - Điểm Log-Score: Spam = {scores['spam']:.3f} | Ham = {scores['ham']:.3f}")
    print(f"  - Dự đoán: {pred_lbl.upper()} | Thực tế: {true_lbl.upper()} -> {status}\n")

# ------------------------------------------------------------------------------
# 5. So sánh với mô hình MLE thuần túy (alpha = 0.0)
# ------------------------------------------------------------------------------
print("=" * 80)
print("SO SÁNH XÁC SUẤT CÓ ĐIỀU KIỆN P(w_i | C_k) GIỮA MLE VÀ MAP")
print("=" * 80)
nb_mle = NaiveBayesClassifier(alpha=0.0)
nb_mle.fit(train_X, train_y)

keywords = ["free", "cash", "prize", "meeting", "report", "project"]
print(f"{'Từ khóa':<12} | {'P_MLE(w|Spam)':<14} | {'P_MAP(w|Spam)':<14} | {'P_MLE(w|Ham)':<14} | {'P_MAP(w|Ham)':<14}")
print("-" * 76)
for kw in keywords:
    p_mle_spam = nb_mle.word_probs['spam'].get(kw, 0.0)
    p_map_spam = nb_model.word_probs['spam'].get(kw, 0.0)
    p_mle_ham = nb_mle.word_probs['ham'].get(kw, 0.0)
    p_map_ham = nb_model.word_probs['ham'].get(kw, 0.0)
    print(f"{kw:<12} | {p_mle_spam:<14.4f} | {p_map_spam:<14.4f} | {p_mle_ham:<14.4f} | {p_map_ham:<14.4f}")

# Vẽ biểu đồ so sánh
x = np.arange(len(keywords))
width = 0.2
plt.figure(figsize=(12, 6), dpi=120)
plt.bar(x - 1.5*width, [nb_mle.word_probs['spam'].get(k, 0) for k in keywords], width, label='MLE - Spam', color='#e74c3c', alpha=0.85)
plt.bar(x - 0.5*width, [nb_model.word_probs['spam'].get(k, 0) for k in keywords], width, label='MAP (Laplace) - Spam', color='#c0392b')
plt.bar(x + 0.5*width, [nb_mle.word_probs['ham'].get(k, 0) for k in keywords], width, label='MLE - Ham', color='#3498db', alpha=0.85)
plt.bar(x + 1.5*width, [nb_model.word_probs['ham'].get(k, 0) for k in keywords], width, label='MAP (Laplace) - Ham', color='#2980b9')

plt.title('So sánh xác suất có điều kiện P(w_i | C_k) giữa MLE và MAP (Laplace Smoothing)', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Các từ khóa đại diện', fontsize=11)
plt.ylabel('Xác suất P(w_i | C_k)', fontsize=11)
plt.xticks(x, keywords, fontsize=11)
plt.legend(fontsize=10)
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()

# Lưu biểu đồ vào file ảnh
plt.savefig('/home/sum/project/ai-notebooks/HTTM/naive_bayes_smoothing.png', dpi=150)
print("\nĐã lưu biểu đồ so sánh vào: /home/sum/project/ai-notebooks/HTTM/naive_bayes_smoothing.png")
