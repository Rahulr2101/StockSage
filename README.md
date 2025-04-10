# Finance Chatbot

A financial assistant chatbot powered by LLaMA and real-time financial data from Yahoo Finance.

## Features

- Real-time stock ticker information
- ESG (Environmental, Social, Governance) scores
- Market data analysis
- Financial advice based on current market conditions
- Intelligent responses powered by the LLaMA language model

## Requirements

- Python 3.8+
- Flask
- llama-cpp-python
- Requests
- Flask-CORS

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/finance-chatbot.git
cd finance-chatbot
```

2. Install dependencies
```bash
pip install flask llama-cpp-python requests flask-cors
```

3. Download the LLaMA model
Download the `finance-chat.Q4_K_M.gguf` model file and place it in the project directory.

## Configuration

The application uses RapidAPI to access Yahoo Finance data. The API keys are included in the code, but you may want to replace them with your own.

## Running the Application

```bash
python app.py
```

The server will start on http://0.0.0.0:8080

## API Endpoints

### Chat Endpoint
- URL: `/chat`
- Method: `POST`
- Request Body:
  ```json
  {
    "message": "Tell me about AAPL stock"
  }
  ```
- Response:
  ```json
  {
    "response": "Apple Inc. (AAPL) is currently trading at $..."
  }
  ```

### Health Check
- URL: `/health`
- Method: `GET`
- Response:
  ```json
  {
    "status": "healthy"
  }
  ```

## How It Works

1. The application parses user queries for stock ticker symbols
2. It fetches real-time market data and ESG scores from Yahoo Finance
3. The data is formatted and passed to the LLaMA model
4. The model generates a relevant response based on the financial data and user query

## Logs

The application logs are stored in `finance_api.log`

## License

[MIT](LICENSE)
