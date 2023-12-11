import numpy as np
import matplotlib.pyplot as plt

from kivy.app import App
from kivy.properties import ObjectProperty
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.properties import StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.popup import Popup
from kivy.uix.label import Label


class MainWidget(BoxLayout):
    pass

class MainWindow(Screen):
    pass

class InfoWindow(Screen):
    pass

class GuideWindow(Screen):
    pass

class GradientProjectionWindow(Screen):
    matrix = ObjectProperty(None)

    def gradient(self, x, f, coef):
        h = 1e-6  # 0.000001
        return (f(x + h, coef) - f(x - h, coef)) / (2 * h)

    def projection(self, x, x_min, x_max):
        return np.clip(x, x_min, x_max)

    def f(self, x, coefficient):
        return coefficient * (x ** 2)

    def getGraph(self):

        text = self.matrix.text.strip()
        array = list(map(float, text.split()))

        coef, x_min, x_max, alpha, max_iter, x0 = array

        # Задаем коэффициенты параболы
        a, b, c = coef, 0, 0

        # Создаем массив значений x от -1 до 1 с шагом 0.01
        x_values = np.arange(x_min, x_max, 0.01)

        # Вычисляем значения y для каждого x
        y_values = a * x_values ** 2 + b * x_values + c

        # Строим график
        plt.plot(x_values, y_values, label=f'y = {a}x^2')
        plt.title(f'y = {a}x^2')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()

        # Устанавливаем пределы для осей x и y
        plt.xlim(x_min - 2, x_max + 2)
        plt.ylim(0, x_max ** 2 + 2)

        # Отображаем ограничения x_min и x_max
        plt.axvline(x=x_min, color='r', linestyle='--', label=f'x_min = {x_min}')
        plt.axvline(x=x_max, color='g', linestyle='--', label=f'x_max = {x_max}')

        # Получаем точку, полученную методом проекции градиента
        projection_result = self.gradientProjection(coef, x_min, x_max, x0, alpha, max_iter)

        # Отображаем точку на графике
        plt.scatter(projection_result, a * projection_result ** 2, color='m', marker='o', label='Projection Result')

        # Добавляем подпись координат к точке
        plt.text(projection_result, a * projection_result ** 2,
                 f'({projection_result:.2f}, {a * projection_result ** 2:.2f})',
                 fontsize=8, ha='right', va='bottom')

        plt.grid(True)
        plt.show()


    def gradientProjection(self, coefficient, x_min, x_max, x0, alpha, max_iter, h=1e-6):
        """
        :param x0: стартовое значение х
        :param alpha: размер шага
        :param h: норма разности между текущим и новым значением х
        :param max_iter: заданное количество итераций
        :return: реализация метода проекции градиента с возвращением найденного оптимального значения
        """
        x = np.array([x0])
        for _ in range(int(max_iter)):
            x_new = x - alpha * self.gradient(x, self.f, coefficient)
            x_projected = self.projection(x_new, x_min, x_max)
            if np.linalg.norm(x_projected - x) < h:
                break
            x = x_projected
        return x[0]

    def getResult(self):
        text = self.matrix.text
        try:
            array = list(map(float, text.split()))

            if len(array) == 6:
                coef, x_min, x_max, alpha, max_iter, x0 = array

                result = self.gradientProjection(coef, x_min, x_max, x0, alpha, max_iter)

                # Display the result in the application window
                result_label = f"Result: ({result:.2f};{self.f(result, coef):.2f}), \nIterations: {max_iter}, \nalpha: {alpha}"
                self.manager.get_screen("result").ids.result_label.text = result_label
                manager = self.manager
                manager.current = "result"
            else:
                self.show_popup("Error", "Please enter exactly SIX values.")
                self.matrix.text = ""
        except ValueError:
            self.show_popup("Error", "Invalid input. Please enter numeric values.")
            self.matrix.text = ""


    def clear_button(self):
        self.matrix.text = ""


    def show_popup(self, title, content):
        popup = Popup(title=title, content=Label(text=content), size_hint=(None, None), size=(400, 400))
        popup.open()


class ResultWindow(Screen):
    result_label = ObjectProperty(None)

    def set_result(self, result):
        self.result_label.text = f"Result: {result}"

    def showGraph(self):
        # Получаем текущий экран
        gradient_projection_window = self.manager.get_screen("diff")

        # Получаем текст из текстового поля matrix
        matrix_text = gradient_projection_window.ids.matrix.text.strip()

        # Проверяем, что в поле есть три значения
        matrix_values = matrix_text.split()
        if len(matrix_values) < 3:
            # Выводим предупреждение
            self.show_popup("Error", "Please enter exactly three values in the matrix field.")
            return

        # Вызываем метод getGraph с передачей параметров
        gradient_projection_window.getGraph()


class WindowManager(ScreenManager):
    with open("information.txt") as info:
        INFO = StringProperty(info.read())
    with open("user_guide.txt") as guide:
        GUIDE = StringProperty(guide.read())


kv = Builder.load_file("app.kv")


class MyMainApp(App):
    title = 'GRADIENT PROJECTION'

    def build(self):
        return kv


if __name__ == "__main__":
    MyMainApp().run()
