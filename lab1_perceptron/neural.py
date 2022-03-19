import numpy as np
import pandas as pd


class Perceptron:
    def __init__(self, input_size, hidden_sizes, output_size):

        self._w_in = np.zeros((1 + input_size, hidden_sizes))
        self._w_in[0, :] = (np.random.randint(0, 3, size=(hidden_sizes)))
        self._w_in[1:, :] = (np.random.randint(-1, 2, size=(input_size, hidden_sizes)))

        self._w_out = np.random.randint(0, 2, size=(1 + hidden_sizes, output_size)).astype(np.float64)
        # self.Wout = np.random.randint(0, 3, size = (1+hiddenSizes,outputSize))
        self._old_weight_array = [self._w_out.tolist()]

    def predict(self, Xp):
        hidden_predict = np.where((np.dot(Xp, self._w_in[1:, :]) + self._w_in[0, :]) >= 0.0, 1, -1).astype(np.float64)
        out = np.where((np.dot(hidden_predict, self._w_out[1:, :]) + self._w_out[0, :]) >= 0.0, 1, -1).astype(
            np.float64)
        return out, hidden_predict

    def _search_for_repetitions(self, min_lenght):
        local_w_out = self._w_out.tolist()
        min_lenght -= 1
        for i in range(len(self._old_weight_array) - min_lenght, len(self._old_weight_array) // 2, -1):
            if str(self._old_weight_array[i:] + [local_w_out])[1:-1] in str(self._old_weight_array[:i])[1:-1]:
                print("Цикл:")
                for i in self._old_weight_array[i:] + [local_w_out]:
                    print(f"{i}")
                return True
        return False

    def train(self, X, y, eta=0.01):
        age = 0
        print(f"Стартовые значения w_out - {self._w_out.reshape(1, -1)}")
        while True:
            print(f"Эпоха {age}")
            errors = 0
            # np.random.shuffle(X)
            for xi, target, j in zip(X, y, range(X.shape[0])):
                pr, hidden = self.predict(xi)
                self._w_out[1:] += ((eta * (target - pr)) * hidden).reshape(-1, 1)
                self._w_out[0] += eta * (target - pr)
                if target != pr:
                    errors += 1
            print(f"Ошибок - {errors / X.shape[0] * 100}%")
            print(f"Значения w_out {self._w_out.reshape(1,-1)}")
            print(f"Конец эпохи {age}\n")
            if errors == 0 or self._search_for_repetitions(2):
                return self
            self._old_weight_array.append(self._w_out.tolist())
            age += 1
