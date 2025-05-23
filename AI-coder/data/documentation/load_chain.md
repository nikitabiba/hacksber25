## Функция `load_chain`

Функция `load_chain` предназначена для загрузки заранее настроенных цепочек обработки из конфигурационного файла или по имени.

### Пример использования

```python
from langchain.chains import load_chain

chain = load_chain("summarize_chain")
response = chain.run("This is a long article text...")
