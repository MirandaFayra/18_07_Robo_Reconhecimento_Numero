#!/usr/bin/env python
# coding: utf-8

# In[3]:


# # KNN 
# # importação das bibliotecas
# import matplotlib.pyplot as plt
# import numpy as np
# %matplotlib inline
# #IMPORT SKLEARN BIBLIOTECA DE AI MCL
# from sklearn.datasets import load_breast_cancer


# In[4]:


# dados = load_breast_cancer()
# print (dados['DESCR'])


# In[6]:


# # TREINAR O ALGORITIMO PARA TREINAR E TESTAR

# #DIVIDIR A BASE PARA TER VARIAVÉIS PARA TESTAR AS RESPOSTAS
# from sklearn.model_selection import train_test_split
# # CRIAR AS VARIAVEIS DE TREINO E TESTE E RECORTAR - retorna uma lista, dividida entre as variáveis 
# # (75% treino aprender e 25% teste)
# xTreino, xTeste, yTreino, yTeste = train_test_split(x,y)


# In[5]:


# x = dados.data # variáveis preditoras
# y = dados.target #variáveis de resposta


# In[ ]:


# xTreino.shape


# In[ ]:


# from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


# # MODELO PARA CLASSIFICAR OS VIZINHOS E CLASSIFICAR O TIPO DE CANCER
# classf= KNeighborsClassifier(n_neighbors= 5)
# #TREINANDO MODELO  COM OS DADOS DE TREINO - FIT -> TREINAR MODELO
# classf.fit(xTreino,yTreino)


# In[ ]:


# # FAZER PREDIÇÃO A PARTIR DOS DADOS E COMPARAR COM A BASE
# preditos= classf.predict(xTeste)


# Performace do Modelo

# In[102]:


from sklearn.metrics import accuracy_score


# In[103]:


# print (accuracy_score(yTeste,preditos))


# Redes Neurais Artificiais para Classificação de Imagens

# In[104]:


from sklearn.datasets import load_digits
digitos=load_digits()
print (digitos['DESCR'])


# In[105]:


# algoritimos de redes neurais para ler qual número é !
digitos.keys()


# In[43]:


# qual digito é cada valor do desenho
digitos.target


# In[44]:


# dentro da [] posição do elemento estudado
dados_teste = digitos.images [29]


# In[45]:


# Array de pixel 
print(dados_teste)


# In[46]:


# plotar a imagem 
plt.imshow(dados_teste,cmap='gray_r')


# In[47]:


digitos.target[500]


# In[48]:


X = digitos.data
Y = digitos.target


# In[106]:


from sklearn.model_selection import train_test_split


# In[118]:


xTreino, xTeste, yTreino, yTeste= train_test_split (X,Y)


# In[119]:


# Importando Rede Neural
from sklearn.neural_network import MLPClassifier


# In[120]:


# Criando o Nosso Modelo
# (50,50) neuronio/camada
rede = MLPClassifier(hidden_layer_sizes=(50,50)) 


# In[ ]:


rede.fit(xTreino, yTreino)


# In[ ]:


preditos = rede.predict(xTeste)


# In[ ]:


print(accuracy_score(yTeste, preditos))


# In[ ]:


yTeste


# In[ ]:


preditos


# In[ ]:


import cv2


# In[ ]:


# # abrindo a imagem
# im = cv2.imread ('digit-reco-1-in.jpg')

# # converter para a escala de cinza
# img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# img_gray= cv2.GaussianBlur(img_gray, (5,5), 0)
# _, img_lim = cv2.threshold(img_gray, 90, 255, cv2.THRESH_BINARY_INV)
# ctrs,hier = cv2.findContours(img_lim.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# rects = [cv2.boundingRect(ctr) for ctr in ctrs]
# # para fazer retangulo e  destacar da lista
# for rect in rects:
#     cv2.rectangle(im, (rect[0], rect[1])), (rect[0] + rect[2], rect[1] + rect [3]) ,(0, 255, 0), 3
    
#     leng =int(rect[3])
#     pt1 = int (rect[1] + rect [3] //2 - leng//2)
#     pt2 = int (rect[0] + rect [2] //2 - leng//2)

# #   recorte dentro de um retângulo
#     roi = img_lim[pt1 : pt1 + leng, pt2 : pt2 - leng]

# #   condição para pegar a imagens válidas

#     if roi.shape[0]> 8 and roi.shape [1] >8:
#         roi = cv2.resize(roi, (8,8), interpolation =cv2.INTER_AREA)
#         roi = roi.flatten()
#         pred = rede.predict([roi])
#     else:
#         pred = ['']
# #  coordenadas x e y do retângulo posicionamento
#     cv2.putText(im, str(pred[0]) , (rect[0]), rect[1]),cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255,0), 3
            
# cv2.imshow('Resultado ', im)

# cv2.waitKey(0)
# cv2.destroyAllWindows


# In[ ]:


# im = cv2.imread('digit-reco-1-in.jpg') #coloca imagem na var
# img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
# img_gray = cv2.GaussianBlur(img_gray, (5,5), 0)
# _, imag_lim = cv2.threshold(img_gray, 90, 255, cv2.THRESH_BINARY_INV) #pegar o limiar e transforma em binário
# ctrs, hier = cv2.findContours(imag_lim.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# rects = [cv2.boundingRect(ctr) for ctr in ctrs]
# for rect in rects:
#     cv2.rectangle(im, (rect[0],rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 255, 0), 3)
#     leng = int(rect[3])
#     pt1 = int(rect[1] + rect[3]//2 - leng//2)
#     pt2 = int(rect[0] + rect[3]//2 - leng//2)
    
#     roi = imag_lim[pt1 : pt1 + leng, pt2 : pt2 + leng]
    
#     if roi.shape[0] > 8 and roi.shape[1] > 8:
#         roi = cv2.resize(roi, (8,8), interpolation=cv2.INTER_AREA)#redimenciona para 8 por 8
#         roi = roi.flatten() #transforma o array
#         pred = rede.predict([roi])
#     else:
#         pred = ['']
#     cv2.putText(im, str(pred[0]), (rect[0], rect[1]), cv2.FONT_HERSHEY_COMPLEX, 2,(0,255,255), 3)
# cv2.imshow('Resultado', im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Aplicando rede neural artificial sobre imagens de faces humanas

# In[7]:


from sklearn.datasets import fetch_lfw_people


# In[8]:


lfw_people = fetch_lfw_people(min_faces_per_person=70, resize= 0.4)


# In[ ]:




