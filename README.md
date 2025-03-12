# Integração e Controle de Qualidade no Serviço de Chatbot para Suporte Rappi

## a) Estrutura de Integração

### 1. Visão Geral
O serviço de chatbot para suporte de entregadores da Rappi utiliza um fluxo de integração baseado na técnica RAG (Retrieval-Augmented Generation), que combina a recuperação de informações de um documento PDF indexado com a geração de respostas via LLM (Large Language Model). A integração envolve diversas camadas, módulos, componentes, serviços e processos, garantindo que o suporte seja rápido, preciso e escalável.

### 2. Camadas e Módulos

#### Camada de Apresentação
- **Interface de Usuário:**  
  - **Terminal (CLI):** Inicialmente, a interação ocorre via terminal, onde o usuário (entregador) digita sua pergunta.
  - *Possível expansão para interfaces web ou mobile futuramente.*

#### Camada de Aplicação
- **Lógica do Chatbot:**  
  - Implementada no arquivo `chatbot.py`.
  - Responsável por orquestrar o fluxo de entrada do usuário, validação, composição de prompt, chamada à API LLM e atualização da memória de conversação.

#### Camada de Dados e Serviços
- **Document Loader e Indexação:**  
  - **PyPDFLoader:** Carrega o documento `chatbot-rag.pdf`.
  - **Divisão de Texto:** Utiliza `RecursiveCharacterTextSplitter` para dividir o PDF em trechos menores.
- **Serviço de Embeddings e Busca:**  
  - **Hugging Face Transformers:** Gera embeddings utilizando o modelo `sentence-transformers/all-MiniLM-L6-v2`.
  - **FAISS:** Indexa os embeddings e permite a busca por similaridade para recuperar os trechos mais relevantes.
- **Serviço de Geração de Resposta (LLM):**  
  - **Groq API:** Utilizado para gerar respostas com base no prompt composto que inclui a consulta do usuário e os trechos recuperados.
  - **LangChain:** Orquestra a integração entre recuperação e geração, utilizando o componente `LLMChain`.
- **Memória de Conversação:**  
  - **ConversationBufferWindowMemory:** Armazena o histórico de interações para manter o contexto entre as perguntas.
- **Monitoramento e Métricas:**  
  - Registro de tempos de resposta, contagem total de interações e cálculo do tempo médio de resposta.

### 3. Componentes, Serviços e Processos

#### Componentes e Serviços
- **Módulos:**
  - `chatbot.py`: Implementa a lógica central do chatbot.
  - `chatbot-rag.pdf`: Documento que contém informações utilizadas na recuperação.
  - `chatbot-tests.py`: Conjunto de testes automatizados para validar o fluxo.
- **Serviços Externos:**
  - **Groq API:** Serviço de LLM para geração de respostas.
  - **FAISS:** Serviço para indexação e recuperação de dados.
  - **Hugging Face Transformers:** Serviço para geração de embeddings.

#### Hardware e Software
- **Hardware:**  
  - Servidores que executam a aplicação Python (podendo ser físicos ou na nuvem).
- **Software:**  
  - **Python 3:** Linguagem de programação.
  - **LangChain:** Biblioteca para integração de LLMs e memória de conversação.
  - **FAISS:** Biblioteca para busca por similaridade.
  - **PyPDFLoader:** Biblioteca para carregamento e extração de PDFs.
  - **dotenv:** Para gerenciamento de variáveis de ambiente.
  - **pytest:** Para execução de testes automatizados.
  - **Logging:** Para registro e monitoramento das operações.

#### Processos de Integração
- **Validação de Entrada:** Verifica se a consulta do usuário é válida (não vazia ou apenas espaços).
- **Indexação de PDF:** Carrega o documento, divide o texto e gera um índice para busca.
- **Recuperação de Documentos:** Utiliza FAISS para buscar os trechos mais relevantes do PDF com base na consulta.
- **Composição do Prompt:** Combina os trechos recuperados com a consulta do usuário para formar o prompt que será enviado ao LLM.
- **Geração de Resposta:** Chama o serviço LLM (Groq API) para gerar uma resposta.
- **Atualização de Memória e Métricas:** Armazena o histórico da conversa e registra o tempo de resposta, o total de interações e outras métricas relevantes.

---

## b) Controle de Qualidade de Integração

### 1. Documentação e Monitoramento

#### Tempos e Protocolos
- **Medição do Tempo de Resposta:**  
  - Cada consulta é temporizada, registrando o tempo de início e fim para calcular o tempo de processamento.
- **Protocolos de Integração:**  
  - O fluxo segue um protocolo documentado, definido por uma versão (por exemplo, "protocol_version": "1.0.0") e registros das versões dos serviços utilizados (Groq API, FAISS, Hugging Face, etc.).
  
#### Versões
- **Versão do Protocolo:**  
  - Documentada no dicionário de metadados (`INTEGRATION_METADATA`), que inclui a versão do protocolo, versão do LLM, versão do FAISS, e a biblioteca de embeddings utilizada.
- **Rastreabilidade de Versões:**  
  - Todas as versões relevantes são registradas em logs, facilitando auditorias e a manutenção do fluxo.

#### Tratamento de Exceções
- **Blocos Try/Except:**  
  - O código utiliza blocos try/except para capturar quaisquer exceções durante o fluxo de integração.
  - Em caso de erro, o sistema registra o stack trace completo e retorna uma mensagem de fallback ao usuário, garantindo que o fluxo não seja interrompido abruptamente.
- **Registro de Erros:**  
  - Utiliza a biblioteca `logging` para registrar todas as exceções e falhas, permitindo que a equipe técnica possa investigar e corrigir problemas com base nos logs gerados.

### 2. Exemplificação de Código para Controle de Qualidade

O código abaixo demonstra como a integração é monitorada e como são registradas as métricas, os tempos de resposta, as versões do protocolo e o tratamento de exceções:

```python
import os
import time
import logging
from dotenv import load_dotenv
from chatbot import RappiDeliveryChatbot

# Carrega variáveis de ambiente (incluindo GROQ_API_KEY)
load_dotenv()

# Configuração do logging para controle de qualidade da integração
logging.basicConfig(
    filename="integration.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Metadados de controle da integração
INTEGRATION_METADATA = {
    "protocol_version": "1.0.0",
    "llm_service_version": "Groq API v1",
    "faiss_version": "FAISS 1.7.1",
    "embedding_library": "HuggingFace Transformers (sentence-transformers/all-MiniLM-L6-v2)"
}

def log_integration_metadata():
    """Registra os metadados de integração para rastreabilidade."""
    logging.info("Metadados de Integração: %s", INTEGRATION_METADATA)

def process_user_query(userQuery: str) -> str:
    """
    Processa a consulta do usuário através de todo o fluxo de integração:
    1. Validação de entrada.
    2. Recuperação de documentos relevantes utilizando RAG.
    3. Geração de resposta via LLM.
    4. Atualização da memória de conversação e registro de métricas.
    """
    chatbot = RappiDeliveryChatbot()
    
    # Registra os metadados no início do fluxo de integração.
    log_integration_metadata()
    
    # Registra o tempo de início
    startTime = time.time()

    try:
        # Processa a consulta. Este método internamente:
        # - Valida a entrada do usuário.
        # - Utiliza FAISS para recuperar trechos relevantes do documento indexado.
        # - Compoe um prompt e envia para a API Groq.
        # - Atualiza a memória de conversação.
        response = chatbot.handle_question(userQuery)
        logging.info("Consulta processada com sucesso: %s", userQuery)
    except Exception as e:
        # Tratamento de exceções: registra o erro com rastreamento.
        logging.exception("Erro ao processar a consulta '%s': %s", userQuery, str(e))
        response = "Desculpe, ocorreu um erro no sistema de suporte. Tente novamente mais tarde."

    # Registra o tempo final e calcula o tempo de processamento.
    endTime = time.time()
    responseTime = endTime - startTime
    logging.info("Tempo de resposta para a consulta '%s': %.3f segundos", userQuery, responseTime)

    # Registra as métricas dessa interação.
    metrics = chatbot.get_metrics()
    logging.info("Métricas após a consulta '%s': %s", userQuery, metrics)

    return response

if __name__ == "__main__":
    # Fluxo interativo de integração
    print("Bem-vindo ao Suporte de Entregas da Rappi!")
    print("Versão do Protocolo:", INTEGRATION_METADATA["protocol_version"])
    print("Digite 'sair' para encerrar.")

    while True:
        userQuery = input("\nDigite sua pergunta: ").strip()
        if userQuery.lower() == "sair":
            print("Encerrando o chatbot. Obrigado!")
            break
        
        # Processa a consulta e exibe a resposta.
        finalResponse = process_user_query(userQuery)
        print("\nResposta do Chatbot:", finalResponse)

