import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

class MLP:
    def inicializacao(self, entradas, c_hide, saida, tx_apendizagem, iteracoes):
        self.entradas = entradas
        self.c_hide = c_hide
        self.saida = saida
        self.tx_apendizagem = tx_apendizagem
        self.iteracoes = iteracoes
        self.pesos_hide = np.random.rand(self.entradas.shape[1], self.c_hide) # Define os pesos da camada escondida com valores aleatorios entre 0 e 1
        self.pesos_saida = np.random.rand(self.c_hide, self.saida.shape[1])
        self.bias_hide = np.random.rand(1, self.c_hide)
        self.bias_saida = np.random.rand(1, self.saida.shape[1])
        
    def ativacao(self, x):
        return 1 / (1 + np.exp(-x)) # A função de ativação funciona retornando valores entre 0 e 1, dependendo do valor de x 
    
    def derivada_ativacao(self, x):
        return x * (1 - x) # É necessário para o cálculo do erro na retropropagação
    
    def treino(self, x, y):
        for epoca in range(self.iteracoes):
            # Propagação
            soma_hide = np.dot(x, self.pesos_hide) + self.bias_hide 
            saida_hide = self.ativacao(soma_hide) 

            soma_saida = np.dot(saida_hide, self.pesos_saida) + self.bias_saida 
            saida_final = self.ativacao(soma_saida)

            # Backprop
            erro_saida = y - saida_final 
            ajuste_saida = erro_saida * self.derivada_ativacao(saida_final) 

            erro_escondido = ajuste_saida.dot(self.pesos_saida.T) 
            ajuste_escondido = erro_escondido * self.derivada_ativacao(saida_hide) 

            # Atualização
            self.pesos_saida += saida_hide.T.dot(ajuste_saida) * self.tx_apendizagem
            self.bias_saida += np.sum(ajuste_saida, axis=0, keepdims=True) * self.tx_apendizagem

            self.pesos_hide += x.T.dot(ajuste_escondido) * self.tx_apendizagem
            self.bias_hide += np.sum(ajuste_escondido, axis=0, keepdims=True) * self.tx_apendizagem

            # Métricas por epoch (é o que ela pediu, so que de 100 em 100)
            if (epoca+1) % 100 == 0:
                loss = np.mean(np.square(y - saida_final))
                acc = np.mean(np.round(saida_final) == y)
                print(f"Epoch {epoca+1}/{self.iteracoes} - loss: {loss:.4f} - acc: {acc:.4f}")

            
    def previsao(self, x):
        soma_hide = np.dot(x, self.pesos_hide) + self.bias_hide
        saida_hide = self.ativacao(soma_hide)
        soma_saida = np.dot(saida_hide, self.pesos_saida) + self.bias_saida
        saida_final = self.ativacao(soma_saida)
        return np.round(saida_final) #np.round arredonda os valores para o inteiro mais próximo (0 ou 1 nesse caso)
            
            
            
            
def prepara_dados(arq):
    # Lê o CSV já trocando vírgula decimal por ponto
    df = pd.read_csv(arq, decimal=',')
    df = df.drop('Animal_ID', axis=1)
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0],inplace=True)
        else:
            df[col].fillna(df[col].median(),inplace=True)
    
    Y = df['Pregnancy_Status'].values
    X = df.drop('Pregnancy_Status', axis=1)
    X = pd.get_dummies(X, drop_first=True) # Converte variáveis categóricas em variáveis dummy
    
    # Identifica as colunas que são numéricas para aplicar a padronização
    numeric_cols = X.select_dtypes(include=np.number).columns

    # Cria o objeto scaler, que serve para padronizar os dados
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    X = X.values.astype(float)
    Y = np.where(Y == 'Yes', 1, 0)

    return X, Y.reshape(-1, 1)

    
X, Y = prepara_dados('dados.csv')

# Divide 70% treino / 30% teste
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42)

mlp = MLP()
mlp.inicializacao(X_train, c_hide=10, saida=Y_train, tx_apendizagem=0.01, iteracoes=5000)
mlp.treino(X_train, Y_train)

y_pred = mlp.previsao(X_test)
print("\n[INFO] avaliando a rede neural (70/30 split)...")
print(classification_report(Y_test, y_pred))
print("Número de previsões corretas:", np.sum(Y_test == y_pred))
print("Número de previsões incorretas:", np.sum(Y_test != y_pred))
print("numero de neuronios em cada camada intermediaria:", mlp.c_hide)
print("Numero de camadas intermediarias:")
