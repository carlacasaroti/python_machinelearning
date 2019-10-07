"""

Neste exemplo vamos utilizar a Naive Bayes.
A abordagem Naive Bayes e baseada no teorema de probabilidade de Bayes
e tem como objetivo calcular a probabilidade que uma amostra desconhecida pertenca
a cada uma das classes possiveis, ou seja, predizer (ou adivinhar) a classe mais provavel.

"""
# importing functions to be used to create the models

from sklearn.naive_bayes import MultinomialNB

# defining the dataset

bichinho1 = [1, 1, 1]  # 1st - is it cute?, 2nd - does it have little ears?, 3rd - does it meow?
bichinho2 = [1, 0, 1]
bichinho3 = [0, 1, 1]
bichinho4 = [1, 1, 0]
bichinho5 = [0, 1, 0]
bichinho6 = [0, 1, 0]
dados = [bichinho1, bichinho2, bichinho3, bichinho4, bichinho5, bichinho6]  # used in the model

# defining the classes : 1 - cat, -1 - dog

marcacoes = [1, 1, 1, -1, -1, -1]  # regarding the entries - for instance, bichinho1 is a cat

# creating model using the library
modelo = MultinomialNB()
modelo.fit(dados, marcacoes)

# making predictions using other dataset
bicho_misterioso1 = [1, 1, 1]
bicho_misterioso2 = [1, 0, 0]
bicho_misterioso3 = [0, 0, 1]
teste = [bicho_misterioso1, bicho_misterioso2, bicho_misterioso3]

# using the trained model
resultado = modelo.predict(teste)

# the right answers dor the test dataset
marcacoes_teste = [1, -1, 1]

# comparing the results from the model and the real answers from the test dataset
print('Resultado: ')
print(resultado)
print ('Marcacoes: ')
print(marcacoes_teste)
