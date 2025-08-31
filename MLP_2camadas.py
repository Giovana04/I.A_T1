import numpy as np # Serve para fazer operações matemáticas
import pandas as pd # Serve para ler o CSV
from sklearn.metrics import classification_report # Serve para mostrar o relatório de classificação
from sklearn.model_selection import train_test_split # Serve para dividir os dados em treino e teste
from sklearn.preprocessing import StandardScaler # Serve para padronizar os dados
import time 

class MLP:
    def __init__(self, tamanhos_das_camadas, tx_apendizagem, iteracoes):
        self.tamanhos_das_camadas = tamanhos_das_camadas
        self.tx_apendizagem = tx_apendizagem
        self.iteracoes = iteracoes
        self.pesos = []
        self.bias = []
        
        for i in range(len(tamanhos_das_camadas) - 1):
            self.pesos.append(np.random.randn(tamanhos_das_camadas[i], tamanhos_das_camadas[i+1]) * 0.1)
            self.bias.append(np.zeros((1, tamanhos_das_camadas[i+1])))

    def ativacao(self, x):
        return 1 / (1 + np.exp(-x))

    def derivada_ativacao(self, x): 
        return x * (1 - x)

    def previsao(self, x):
        ativacoes = x
        for i in range(len(self.pesos)):
            soma_ponderada = np.dot(ativacoes, self.pesos[i]) + self.bias[i]
            ativacoes = self.ativacao(soma_ponderada)
        return np.round(ativacoes)

    def treino(self, x, y):
        stard_time = time.perf_counter()
        for epoca in range(self.iteracoes or loss < 0.00001):
            lista_ativacoes = [x]
            ativacoes_camada = x
            
            for i in range(len(self.pesos)):
                soma_ponderada = np.dot(ativacoes_camada, self.pesos[i]) + self.bias[i] 
                ativacoes_camada = self.ativacao(soma_ponderada)
                lista_ativacoes.append(ativacoes_camada)
                
            saida_final = lista_ativacoes[-1]

            # --- Início do Retropropagação ---
            erro = y - saida_final
            
            for i in reversed(range(len(self.pesos))):
                derivada = self.derivada_ativacao(lista_ativacoes[i+1])
                delta = erro * derivada
                erro = delta.dot(self.pesos[i].T)
                
                self.pesos[i] += lista_ativacoes[i].T.dot(delta) * self.tx_apendizagem # Como funciona: https://www.youtube.com/watch?v=tIeHLnjs5U8
                self.bias[i] += np.sum(delta, axis=0, keepdims=True) * self.tx_apendizagem
            
            # A cada 100 ciclos, imprime o status do treino
            if (epoca + 1) % 100 == 0:
                loss = np.mean(np.square(y - saida_final)) # Mede o erro 
                acc = np.mean(np.round(saida_final) == y) # Mede a acurácia
                print(f"Ciclo {epoca+1}/{self.iteracoes} -> loss: {loss:.5f} - acc: {acc:.5f}")
                if loss < 0.00001: break
                
        end_time = time.perf_counter()
        tempo = end_time - stard_time
        print(f"\nTempo de treino: {tempo:.2f} segundos")
                


def prepara_dados_com_scaler(arq):
    df = pd.read_csv(arq, decimal=',')
    df = df.drop('Animal_ID', axis=1)

    # Arruma os valores ausntes
    for column in df.columns:
        if df[column].dtype == 'object':
            # Coloca o resultado de volta na coluna
            df[column] = df[column].fillna(df[column].mode()[0]) # Usa a moda para colunas categóricas
        else:
            # Coloca o resultado de volta na coluna
            df[column] = df[column].fillna(df[column].median()) # Usa a mediana para colunas numéricas

    Y = df['Pregnancy_Status'].values
    X = df.drop('Pregnancy_Status', axis=1)
    
    X = pd.get_dummies(X, drop_first=True) # Converte colunas categóricas em numéricas, então cada coluna com mais de 1 valor vira várias colunas binárias
    
    numeric_cols = X.select_dtypes(include=np.number).columns
    scaler = StandardScaler() # Padroniza os dados numéricos, se tiver valores muito altos ou muito baixos, pode atrapalhar o treino então ele padroniza para os pesos não dominarem um ou outro
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols]) 

    X = X.values.astype(float)
    Y = np.where(Y == 'Yes', 1, 0)
    
    return X, Y.reshape(-1, 1)

# ---- Main ----

X, Y = prepara_dados_com_scaler('dados.csv')
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.3, random_state=42)

n_col_entrada = X_treino.shape[1] # Numéro de características de entrada, que são as colunas do dataset (já modificadas com get_dummies)
n_camada_oculta_1 = 10
n_camada_oculta_2 = 5
n_camada_saida = 1

tamanhos_das_camadas = [n_col_entrada, n_camada_oculta_1, n_camada_oculta_2, n_camada_saida]

mlp = MLP(tamanhos_das_camadas=tamanhos_das_camadas, tx_apendizagem=0.1, iteracoes=15000) # tx_apendizagem é a taxa de aprendizagem, que define o quanto os pesos serão ajustados a cada iteração, 0.5 não funciona porque é muito alto
mlp.treino(X_treino, Y_treino)

previsoes = mlp.previsao(X_teste)

print("\nTestes:")
print(classification_report(Y_teste, previsoes))
print("Número de previsões corretas:", np.sum(Y_teste == previsoes))
print("Número de previsões incorretas:", np.sum(Y_teste != previsoes))
print("\nTotal de previsões:", len(Y_teste))
print("Número de amostras de teste:", X_teste.shape[0])
print("Número de amostras de treino:", X_treino.shape[0])
print("\nNúmero de neurônios na camada de entrada:", n_col_entrada)
print("Número de neurônios na primeira camada oculta:", n_camada_oculta_1)
print("Número de neurônios na segunda camada oculta:", n_camada_oculta_2)
