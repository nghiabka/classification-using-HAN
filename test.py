from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
from keras_preprocessing.sequence import pad_sequences
from nltk import sent_tokenize

from keras_han.model import HAN
import pickle
import numpy as np


MAX_WORDS_PER_SENT = 120
MAX_SENT = 25
MAX_VOC_SIZE = 300600
GLOVE_DIM = 400

embedding_matrix = pickle.load(open("./model/embedding_matrix.pkl", "rb"))
word_tokenizer = pickle.load(open("./model/word_tokenizer.pkl", "rb"))
han_model = HAN(
    MAX_WORDS_PER_SENT, MAX_SENT, 10, embedding_matrix,
    word_encoding_dim=400, sentence_encoding_dim=200
)
han_model.load_weights("./model/model.hdf5")

text = " Truy tố 1 đối tượng về tội “Vu khống” (NLĐ)- VKSND TPHCM vừa hoàn tất cáo trạng," \
       " chuyển hồ sơ truy tố Dương Kim Khải (sinh năm 1956, ngụ Bình Thạnh) ra trước TAND TPHCM để xét xử về tội “Vu khống”," \
       " do liên quan đến việc giải tỏa để xây dựng trường học Đinh Bộ Lĩnh, tại phường 26, quận Bình Thạnh – TPHCM.Theo cáo trạng," \
       " tháng 4-1997, ông Khải mua (giấy tay) một căn nhà lá (diện tích 5 m x 9 m) với giá 1,3 triệu đồng, " \
       "trên phần đất công lấn chiếm trái phép của dự án. Vào ngày 7-1-2004, ông Nguyễn Quốc Hùng, lúc này là quyền Chủ tịch UBND quận Bình Thạnh, " \
       "đã ra quyết định xử phạt hành chính buộc tháo dỡ các căn nhà xây dựng trên đất lấn chiếm (đợt 1) đối với 24 hộ, trong đó có hộ ông Khải." \
       " Đến ngày 28-1-2004, ông Khải lấy danh nghĩa của tổ dân phố viết đơn khiếu nại ông Nguyễn Quốc Hùng “ra quyết định xử phạt vi phạm hành chính vô đạo đức và trái pháp luật”." \
       " Sau đó, ông Khải kêu gọi những người trong diện giải tỏa ký vào đơn và lôi kéo nhiều người khiếu kiện tập thể." \
       " Trước những chứng cứ của cơ quan điều tra, ông Khải thừa nhận mình đã có hành vi vu khống."

x_test = [text]
X_test_num = np.zeros((len(x_test), MAX_SENT, MAX_WORDS_PER_SENT), dtype='int32')

for i, review in enumerate(x_test):
    sentences = sent_tokenize(review)
    tokenized_sentences = word_tokenizer.texts_to_sequences(
        sentences
    )
    tokenized_sentences = pad_sequences(
        tokenized_sentences, maxlen=MAX_WORDS_PER_SENT
    )

    pad_size = MAX_SENT - tokenized_sentences.shape[0]

    if pad_size < 0:
        tokenized_sentences = tokenized_sentences[0:MAX_SENT]
    else:
        tokenized_sentences = np.pad(
            tokenized_sentences, ((0, pad_size), (0, 0)),
            mode='constant', constant_values=0
        )

    # Store this observation as the i-th observation in
    # the data matrix
    X_test_num[i] = tokenized_sentences[None, ...]

result = han_model.predict(X_test_num)
a =np.argmax(result[0])
print(a)
print(result[0][a]*100)

