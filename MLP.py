import numpy as np
import pandas as pd

class MLP:
    def inicializacao(self, entradas, c_hide, saida, tx_apendizagem, iteracoes):
        self.entradas = entradas
        self.c_hide = c_hide
        self.saida = saida
        self.tx_apendizagem = tx_apendizagem
        self.iteracoes = iteracoes
        self.pesos_hide = np.random.rand(self.entradas.shape[1], self.c_hide) #define os pesos da camada escondida com valores aleatorios entre 0 e 1
        self.pesos_saida = np.random.rand(self.c_hide, self.saida.shape[1])
        self.bias_hide = np.random.rand(1, self.c_hide)
        self.bias_saida = np.random.rand(1, self.saida.shape[1])
        
    def ativacao(self, x):
        return 1 / (1 + np.exp(-x)) # A função de ativação funciona retornando valores entre 0 e 1, dependendo do valor de x 
    
    def derivada_ativacao(self, x):
        return x * (1 - x) # É necessário para o cálculo do erro na retropropagação
    
    def treino(self, x, y):
        for _ in range(self.iteracoes):
            
            # Da entrada a camada escondida - Fazendo a multiplicação dos pesos pelos valores de entrada e somando o bias porque bias é tipo um ajuste extra
            soma_hide = np.dot(x, self.pesos_hide) + self.bias_hide 
            saida_hide = self.ativacao(soma_hide) 
            
            # Da camada escondida a camada de saida -
            soma_saida = np.dot(saida_hide, self.pesos_saida) + self.bias_saida 
            saida_final = self.ativacao(soma_saida)
            
            #calculo do erro da camada de saida
            erro_saida = y - saida_final 
            ajuste_saida = erro_saida * self.derivada_ativacao(saida_final) 
            erro_escondido = ajuste_saida.dot(self.pesos_saida.T) 
            ajuste_escondido = erro_escondido * self.derivada_ativacao(saida_hide) 
        
            
            
            
            
            
def prepara_dados(arq):
    df = pd.read_csv(arq)
    df = df.drop('Animal_ID', axis=1) # Remove a coluna Animal_ID
    X = df.drop('Pregnancy_Status', axis=1).values # Todas as colunas menos a do resultado
    Y = df['Pregnancy_Status'].values # Apenas a coluna do resultado
    Y = np.where(Y == 'Yes', 1, 0) # Converte 'Yes' para 1 e 'Not' para 0
    return X, Y.reshape(-1, 1) # Retorna X e Y como arrays numpy (obs: numpy é tipo array mas mais facil de mexer)
    