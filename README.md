#  Projeto - Conhecendo o AWS SageMaker para tarefas de Machine Learning

<h1>
    <img src="https://i.ibb.co/w6NGdV6/Certificado.jpg" alt="SQL Server e XML-Persistindo XML de maneira eficiente" border="0">
</h1>
<br>
Repositório de códigos da plataforma de Projetos PRO da <a href="https://web.digitalinnovation.one/labs">***Digital Innovation One***</a> sobre AWS Sagemaker
<br>
AWS Sagemaker é um serviço totalmente gerenciado da AWS, que pode nos ajudar a preparar, criar, treinar e implantar modelos de Machine Learning (ML) de alta qualidade rapidamente. Para isso, o expert apresenta algumas das features disponíveis nessa ferramenta, explorando na prática um pouco do seu potencial.
<br>
<h2>
    <img src="https://i.ibb.co/7Qd6fn9/1.png" alt="AWS Sagemaker" border="0">
</h2>
<br>
<h2>
    <img src="https://i.ibb.co/jbPWQCV/3.png" alt="AWS Sagemaker" border="0">
</h2>

# Passo a Passo

<h2>
      <img src="https://i.ibb.co/rvxPmz9/4.png" alt="AWS Sagemaker" border="0">
</h2>

## Passo 1 - Crie uma instância de notebook Amazon SageMaker para preparação de dados
Nesta etapa, cria-se a instância do notebook que usa para baixar e processar seus dados. Como parte do processo de criação, você também cria uma função de gerenciamento de identidade e acesso (IAM) que permite ao Amazon SageMaker acessar dados no Amazon S3.

- Faça login no console do Amazon SageMaker e, no canto superior direito, selecione sua região AWS preferida. Este tutorial usa a região US West (Oregon).
- No painel de navegação esquerdo, escolha ```Notebook instances```, e ```Create notebook instance```.
- Na página ```Create notebook instance```, na caixa ```Notebook instance setting```, preencha os seguintes campos:
  - Para ```Notebook instance name```, digite ```SageMaker-DIO-Live```.
  - Para ```Notebook instance type```, escolha ```ml.t2.medium```.
  - Para ```Elastic inference```, mantenha a seleção padrão de ```none```.
- Na seção ```Permissions and encryption```, para  ```IAM role```, escolha ```Create a new role``` e, na caixa de diálogo ```Create an IAM role```, selecione ```Any S3 bucket``` e escolha ```Create role```.
Observação: se você já tem um bucket que gostaria de usar, escolha ```Specific S3 buckets``` e especifique o nome do bucket.
- Amazon SageMaker creates the ```AmazonSageMaker-ExecutionRole-***``` role.
-  Mantenha as configurações padrão para o restante das opções e selecione ```Create notebook instance```.
Em ```Notebook instances section```, a nova instância do Notebook será mostrada no status de ```Pending```. O Notebook estará disponível quando o status mudar para  ``` InService```. 

## Passo 2 - Preparar os dados
Nesta etapa, você usa sua instância de notebook do Amazon SageMaker para pré-processar os dados de que precisa para treinar seu modelo de aprendizado de máquina e, em seguida, fazer upload dos dados para o Amazon S3.

 - Depois da sua instância do Notebook mudar o status para ```InService``` selecion ```Open Jupyter```
 - Em ```Jupyter``` selecione ```New``` e escolha ```conda_python3```
 - Em uma nova célula de código no Jupyter Notebook, copie e cole o seguinte código e selecione ```Run```
 ```
 # import libraries
import boto3, re, sys, math, json, os, sagemaker, urllib.request
from sagemaker import get_execution_role
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image
from IPython.display import display
from time import gmtime, strftime
from sagemaker.predictor import csv_serializer

# Define IAM role
role = get_execution_role()
prefix = 'sagemaker/DEMO-xgboost-dm'
my_region = boto3.session.Session().region_name # set the region of the instance

# this line automatically looks for the XGBoost image URI and builds an XGBoost container.
xgboost_container = sagemaker.image_uris.retrieve("xgboost", my_region, "latest")

print("Success - the MySageMakerInstance is in the " + my_region + " region. You will use the " + xgboost_container + " container for your SageMaker endpoint.")
 ```
- Crie o bucket S3 para armazenar eus dados. Copie e cole o código a seguir em uma nova célula de código, altere o nome do bucket e selecione ```Run```
```
bucket_name = 'your-s3-bucket-name' # <--- CHANGE THIS VARIABLE TO A UNIQUE NAME FOR YOUR BUCKET
s3 = boto3.resource('s3')
try:
    if  my_region == 'us-east-1':
      s3.create_bucket(Bucket=bucket_name)
    else: 
      s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={ 'LocationConstraint': my_region })
    print('S3 bucket created successfully')
except Exception as e:
    print('S3 error: ',e)
 ```
- Faça o donwload dos dados para a sua instância do SageMaker e carregue os dados em um dataframe. Copie e cole o seguinte código em uma nova célula de código e clique em ```Run```
```
try:
  urllib.request.urlretrieve ("https://d1.awsstatic.com/tmt/build-train-deploy-machine-learning-model-sagemaker/bank_clean.27f01fbbdf43271788427f3682996ae29ceca05d.csv", "bank_clean.csv")
  print('Success: downloaded bank_clean.csv.')
except Exception as e:
  print('Data load error: ',e)

try:
  model_data = pd.read_csv('./bank_clean.csv',index_col=0)
  print('Success: Data loaded into dataframe.')
except Exception as e:
    print('Data load error: ',e)
 ```
 - Misture e divida os dados em dados de treinamento e dados de teste. Copie e cole o código a seguir na próxima célula de código e escolha Executar. Os dados de treinamento (70% dos clientes) são usados durante o loop de treinamento do modelo. Use a otimização baseada em gradiente para refinar iterativamente os parâmetros do modelo. A otimização baseada em gradiente é uma maneira de encontrar os valores dos parâmetros do modelo que minimizam o erro do modelo, usando o gradiente da função de perda do modelo. Os dados de teste (restantes 30% dos clientes) são usados para avaliar o desempenho do modelo e medir quão bem o modelo treinado generaliza para dados invisíveis.

 ```
train_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data))])
print(train_data.shape, test_data.shape)
 ```
 ## Passo 3 - Treinar o modelo de ML
 Nesta etapa, você usa seu conjunto de dados de treinamento para treinar seu modelo de aprendizado de máquina.
 
 - Em uma nova célula de código em seu Notebook Jupyter, copie e cole o código a seguir e escolha Executar. Este código reformata o cabeçalho e a primeira coluna dos dados de treinamento e, em seguida, carrega os dados do bucket S3. Esta etapa é necessária para usar o algoritmo XGBoost pré-construído do Amazon SageMaker.

```
pd.concat([train_data['y_yes'], train_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('train.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket_name, prefix), content_type='csv')
```

- Configure a sessão do Amazon SageMaker, crie uma instância do modelo XGBoost (um estimador) e defina os hiperparâmetros do modelo. Copie e cole o seguinte código na próxima célula de código e escolha ```Run```
```
sess = sagemaker.Session()
xgb = sagemaker.estimator.Estimator(xgboost_container,role, instance_count=1, instance_type='ml.m4.xlarge',output_path='s3://{}/{}/output'.format(bucket_name, prefix),sagemaker_session=sess)
xgb.set_hyperparameters(max_depth=5,eta=0.2,gamma=4,min_child_weight=6,subsample=0.8,silent=0,objective='binary:logistic',num_round=100)
```
- Comece o trabalho de treinamento. Copie e cole o código a seguir na próxima célula de código e escolha Executar. Este código treina o modelo usando a otimização de gradiente em uma instância ml.m4.xlarge. Depois de alguns minutos, você deve ver os registros de treinamento sendo gerados em seu Notebook Jupyter.

```
xgb.fit({'train': s3_input_train})
```

## Passo 4 - Publicar o modelo de ML
Nesta etapa, você implanta o modelo treinado em um endpoint, reformata e carrega os dados CSV e, em seguida, executa o modelo para criar previsões.

- Em uma nova célula de código do Notebook Jupyter, copie e cole o código a seguir e escolha ```Run```. Este código implanta o modelo em um servidor e cria um endpoint SageMaker para acessi. Esta etapa pode levar alguns minutos para ser concluída.
```
xgb_predictor = xgb.deploy(initial_instance_count=1,instance_type='ml.m4.xlarge')
```
- Para realizar a predição dos clientes que irão aderir ao produto do banco ou não na amostra de testes, copie e cole o seguinte código em uma nova célula ```Run```
```
from sagemaker.serializers import CSVSerializer

test_data_array = test_data.drop(['y_no', 'y_yes'], axis=1).values #load the data into an array
xgb_predictor.serializer = CSVSerializer() # set the serializer type
predictions = xgb_predictor.predict(test_data_array).decode('utf-8') # predict!
predictions_array = np.fromstring(predictions[1:], sep=',') # and turn the prediction into an array
print(predictions_array.shape)
```

## Passo 5 - Avaliar a performance do modelo treinado

Em uma nova célula do Notebook Jupyter, copie e cole o seguinte código e selecione ```Run```. Este código compara os valores atuais com os preditos em uma tabela chamada *Matriz de Confusão*. Baseado na predição, pode-se concluir que um cliente irá se inscrever para um certificado de depósito com acurácia de 90% para os clientes dos dados de teste, uma precisão de 65% para os que irão se inscrever e 90% para os que não irão se inscrever.

```
cm = pd.crosstab(index=test_data['y_yes'], columns=np.round(predictions_array), rownames=['Observed'], colnames=['Predicted'])
tn = cm.iloc[0,0]; fn = cm.iloc[1,0]; tp = cm.iloc[1,1]; fp = cm.iloc[0,1]; p = (tp+tn)/(tp+tn+fp+fn)*100
print("\n{0:<20}{1:<4.1f}%\n".format("Overall Classification Rate: ", p))
print("{0:<15}{1:<15}{2:>8}".format("Predicted", "No Purchase", "Purchase"))
print("Observed")
print("{0:<15}{1:<2.0f}% ({2:<}){3:>6.0f}% ({4:<})".format("No Purchase", tn/(tn+fn)*100,tn, fp/(tp+fp)*100, fp))
print("{0:<16}{1:<1.0f}% ({2:<}){3:>7.0f}% ({4:<}) \n".format("Purchase", fn/(tn+fn)*100,fn, tp/(tp+fp)*100, tp))
```

## Passo 6 - Limpar os recursos
Neste passo você irá limpar o ambiente com os recursos
Importante: Encerrar recursos que não estão sendo usados ativamente reduz custos e é uma prática recomendada. Não encerrar seus recursos resultará em cobranças em sua conta.
 - Deletar o seu endpoind: No seu Notebook Jupyter, copie e cole o seguinte código e escolhe ```Run```
 ```
 xgb_predictor.delete_endpoint(delete_endpoint_config=True)
 ```
 - Deletar os artefatos de treino e o bucket S3: No seu Notebook Jupyter, copie e cole o seguinte código e selecione ```Run```
 ```
bucket_to_delete = boto3.resource('s3').Bucket(bucket_name)
bucket_to_delete.objects.all().delete()
 ```
- Excluir o seu SageMaker Notebook: Parar e excluir o seu SageMaker Notebook
  - Abrir o ```SageMaker Console```
  - Em ```Notebook``` escolha ```Notebook instances```
  - Selecione a instância do Notebook criada, selecione ```Actions``` e ```Stop```. Este procedimento pode levar alguns minutos, e quando o status mudar para ```Stopped```, vá para o passo seguinte
  - Selecione ```Actions``` e depois ```Delete```
  - Selecione ```Delete```

## Me siga nas redes sociais

🧑🏼‍💻👩🏼‍💻 https://linktr.ee/ygtecnologia 
<br>
<br> 
Investir em conhecimento rende sempre os melhores juros. Benjamim Franklin
<br>
<br> 
🙏 Oração ! Foco ! Ação ! Yeshua Hamashia