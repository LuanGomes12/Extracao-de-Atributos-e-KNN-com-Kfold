% ********** Luan Gomes Magalhães Lima - 473008                      ********** 
% ********** Tópicos Especiais em Telecomunicações 1 - Prática 2     **********

% Inicializações
clear all;
close all;
clc;

%% Ánalise exploratória e escolha dos atributos 
% Carregar a base de dados
load("InputData.mat");
load("OutputData.mat");

% Matriz de entrada
X = InputData';
% Matriz de saída
y = OutputData;

% Atributos a serem analisados
% 1º Atributo: Média
media = mean(X, 2);

% 2º Atributo: Kurtose
kurtose = kurtosis(X, 0, 2);

% 3º Atributo: Assimetria
assimetria = skewness(X, 0, 2);

% 4º Atributo: Raíz Quadrada do Erro Quadrático Médio
rms = rms(X, 2);

% 5º Atributo: Desvio Padrão
desvio_padrao = std(X, 1, 2);

% 6º Atributo: Variância
variancia = var(X, 0, 2);

% 7º Atributo: Mediana
mediana = median(X, 2);

% 8º Atributo: Valor Mínimo
valor_min = min(X,[], 2);

% 9º Atributo: Valor Máximo
valor_max = max(X, [], 2);

% 10º Atributo: Entropia
entropia = zeros(120, 1);
for i = 1:120
    entropia(i, 1) = entropy(X(i, :));
end

% 11º Atributo: Moda
moda = zeros(120, 1);
for i = 1:120
    moda(i, 1) = mode(X(i, :));
end

energia = zeros(120,1);
potencia = zeros(120,1);
for i = 1:120
    amostra_atual = X(i, :);

    % 12º Atributo: Energia
    energia(i) = sum(abs(amostra_atual).^2);

    % 13º Atributo: Potência
    potencia(i) = mean(abs(amostra_atual).^2);
end

% Atributos escolhidos: Assimetria, Desvio Padrão, Entropia, Média,
% Mediana, Moda, Potência, Valor Máximo, Valor Mínimo, Variância
X_tratado = [assimetria, desvio_padrao, entropia, media, mediana, moda, potencia, valor_max, valor_min, variancia];


% Cores representando cada classe -> Vermelho: -1 e Azul: 1
paleta_cores = [0 0 1; 1 0 0];
cores = paleta_cores((y == -1) + 1, :);

% Gráficos de dispersão dos 10 atributos escolhidos
figure;
scatter(assimetria, 0, [], cores, 'filled');
title("Assimetria");

figure;
scatter(desvio_padrao, 0, [], cores, 'filled');
title("Desvio Padrão");

% figure;
% scatter(energia, 0, [], cores, 'filled');
% title("Energia");

figure;
scatter(entropia, 0, [], cores, 'filled');
title("Entropia");

% figure;
% scatter(kurtose, 0, [], cores, 'filled');
% title("Kurtose");

figure;
scatter(media, 0, [], cores, 'filled');
title("Média");

figure;
scatter(mediana, 0, [], cores, 'filled');
title("Mediana");

figure;
scatter(moda, 0, [], cores, 'filled');
title("Moda");

figure;
scatter(potencia, 0, [], cores, 'filled');
title("Potência");

% figure;
% scatter(rms, 0, [], cores, 'filled');
% title("RMS");

figure;
scatter(valor_max, 0, [], cores, 'filled');
title("Valor Máximo");

figure;
scatter(valor_min, 0, [], cores, 'filled');
title("Valor Mínimo");

figure;
scatter(variancia, 0, [], cores, 'filled');
title("Variância");

%% Criação do classificador
% Features e labels que serão utilizadas no KNN
features = X_tratado;
labels = y;

% Divisão da base em K partes para implementação do K-Fold
K_fold = 10;
num_amostras_kfold = size(features, 1)/K_fold;

% Garantir a aleatoriedade dos índices
ind_embaralhado = randperm(size(features,1));
features = features(ind_embaralhado, :);
labels = labels(ind_embaralhado);

% Matriz para armazenar as acurácias de cada K parte do K-Fold
acuracia = zeros(K_fold, 1);

% Loop do K-Fold
for j = 1 : K_fold
    % Divisão dos índices em teste e treino
    ind_teste = ind_embaralhado((j-1)*num_amostras_kfold+1 : j*num_amostras_kfold);
    ind_treino = setdiff(ind_embaralhado, ind_teste);

    % Conjunto de treinamento
    X_train = features(ind_treino, :);
    y_train = labels(ind_treino);

    % Conjunto de teste
    X_test = features(ind_teste, :);
    y_test = labels(ind_teste);

    % Implementação do KNN
    % Escolha dos K vizinhos mais próximos
    K = 5;

    % Armazena os acertos do classificador KNN
    matriz_acertos = zeros(size(X_test(:, 1)));
    
    % Loop através de cada amostra de teste
    for i = 1:size(X_test, 1)

        % Amostra de teste atual
        amostra_teste = X_test(i, :);
    
        % Calcular a distância euclidiana entre a amostra de teste e as amostras de treinamento
        distancias = pdist2(amostra_teste, X_train, 'euclidean');
    
        % Obter os k vizinhos mais próximos
        [distancia_ordenada, ind_min] = sort(distancias);
        
        % K vizinhos mais próximos
        k_vizinhos = ind_min(1 : K);

        % Classes dos K vizinhos mais próximos
        classes_vizinhos = y_train(k_vizinhos);
    
        % Determinar a classe de cada amostra de teste com base nas classes 
        % dos vizinhos mais próximos -> Previsão
        predict = mode(classes_vizinhos);
    
        % Avaliar o desempenho do classificador comparando com a classe real
        classe_real = y_test(i);
    
        % Verifica a quantidade de acertos do classificador com base em
        % cada amostra de teste
        if predict == classe_real
            matriz_acertos(i, 1) = 1; 
        end

    end

    % Quantidade de acertos em cada K-Fold 
    total_acertos = 0;
    for i = 1 : size(matriz_acertos(:, 1))
        if matriz_acertos(i, 1) == 1
            total_acertos = total_acertos + 1;
        end
    end
     
    % Acurácia em cada K-Fold
    qtd_total_amostras = size(y_test);
    acuracia(j) = total_acertos/qtd_total_amostras(1,1);
end

% Acurácia final do classificador KNN
acuracia_final = mean(acuracia);
fprintf("Acurácia do classificaor: %.2f", acuracia_final*100);