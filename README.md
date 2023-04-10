# Genetic_AI
Данная генетическая нейросеть управляет "бактериями", цель которых - постоянно двигаться. Если бактерия начнет стоять на месте или выйдет за границы экрана, то она умрет. То есть основная цель данной генетической нейросети - продержать бактерию как можно дольше живой.

Сам интеллект подключается с помощью отдельно написанного модуля `AI`\
Всего входных значений 4. Каждое из них - нормализованное расстояние бактерии до каждой из стен. На выходе нейросеть даст вероятности движения бактерии на каждую из четырех сторон, в дальнейшем преобразованные функцией **argmax**

Функция активации нейронов скрытого слоя - **ReLU**\
Функция активации нейронов выходного слоя - **softmax**\
Весов на первом слое - 20 (входной слой - 4 нейрона, скрытый - 5)\
Весов на втором слое - 20 (скрытый слой - 5 нейронов, выходной - 4)


![image](https://user-images.githubusercontent.com/120571667/230972669-183c9464-e88e-45a9-a059-5e12e01b2103.png)


**Итерации, необходимые для полного обучения (пример)**
![image](https://user-images.githubusercontent.com/120571667/230986744-f6a8310a-baf8-4684-9f0b-3dc6ace5cb18.png)
