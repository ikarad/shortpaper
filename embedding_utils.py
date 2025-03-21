
import numpy as np
from pythainlp import word_vector

# โหลดโมเดล Thai2Fit
model = word_vector.WordVector(model_name="thai2fit_wv").get_model()

# ฟังก์ชันสำหรับสร้าง document embedding จาก token list
def document_embedding(tokens):
    valid_tokens = [word for word in tokens if word in model.key_to_index]
    if not valid_tokens:
        return np.zeros(model.vector_size)
    return np.mean(model[valid_tokens], axis=0)
