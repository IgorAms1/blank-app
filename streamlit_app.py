import json
from typing import Dict, Any, List, Coroutine
import asyncio
import streamlit as st
from openai import AsyncOpenAI
from statistics import mean
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Функция для получения API ключа
def get_api_key():
    input_method = st.radio("Выберите метод ввода API ключа:", 
                            ("Ввести вручную", "Использовать сохраненный ключ"))
    
    if input_method == "Ввести вручную":
        api_key = st.text_input("Введите ваш API ключ OpenAI:", type="password")
        if api_key:
            st.button("Сохранить API ключ", on_click=lambda: st.session_state.update({"api_key": api_key}))
    else:
        api_key = st.session_state.get("api_key", "")
        if not api_key:
            st.warning("Сохраненный API ключ не найден. Пожалуйста, введите ключ вручную.")
            api_key = st.text_input("Введите ваш API ключ OpenAI:", type="password")
            if api_key:
                st.button("Сохранить API ключ", on_click=lambda: st.session_state.update({"api_key": api_key}))
    
    return api_key

# Получаем API ключ
api_key = get_api_key()

# Настройка клиента OpenAI
@st.cache_resource
def get_openai_client(api_key):
    return AsyncOpenAI(api_key=api_key)

if api_key:
    client = get_openai_client(api_key)
else:
    st.warning("Пожалуйста, введите API ключ OpenAI для продолжения.")
    st.stop()

async def create_mnemonic(word: str, prompt: str) -> Dict[str, Any]:
    try:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": word}
            ]
        )
        content = response.choices[0].message.content
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"association": content, "meaning": "", "prompt": ""}
    except Exception as e:
        logging.error(f"Ошибка при создании мнемоники для '{word}': {str(e)}")
        return {"association": "Ошибка при создании мнемоники", "meaning": "", "prompt": ""}

async def rate_memory(association: str) -> int:
    try:
        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a memory expert. Rate the following mnemonic association on a scale from 1 to 100 based on how easy it is to remember. Return only the numeric score."},
                {"role": "user", "content": association}
            ]
        )
        return int(response.choices[0].message.content)
    except ValueError:
        logging.warning(f"Не удалось преобразовать оценку в число для ассоциации: {association[:50]}...")
        return 0
    except Exception as e:
        logging.error(f"Ошибка при оценке запоминаемости: {str(e)}")
        return 0

async def process_word(word: str, prompts: List[str]) -> Dict[str, Any]:
    word_results = []
    for prompt in prompts:
        mnemonic = await create_mnemonic(word, prompt)
        association = mnemonic.get('association', '')
        score = await rate_memory(association)
        mnemonic['score'] = score
        word_results.append(mnemonic)
    return {"word": word, "mnemonics": word_results}

def run_async(coroutine: Coroutine[Any, Any, Any]) -> Any:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coroutine)
    finally:
        loop.close()

@st.cache_data
def cached_process_words(words: List[str], prompts: List[str]) -> List[Dict[str, Any]]:
    return [run_async(process_word(word, prompts)) for word in words]

st.title('Мнемоническая ассоциация и оценка запоминаемости')

# Ввод слов пользователем
words_input = st.text_area("Введите слова (по одному на строку):", height=150)
words = [word.strip() for word in words_input.split('\n') if word.strip()]

# Ввод промптов пользователем
st.subheader("Введите промпты:")
num_prompts = st.number_input("Количество промптов", min_value=1, value=1, step=1)
prompts = []
for i in range(num_prompts):
    prompt = st.text_area(f"Промпт {i+1}", height=100)
    prompts.append(prompt)

if st.button('Генерировать мнемоники и оценки'):
    if not words:
        st.error("Пожалуйста, введите хотя бы одно слово.")
    elif not all(prompts):
        st.error("Пожалуйста, заполните все промпты.")
    else:
        with st.spinner('Обработка...'):
            results = cached_process_words(words, prompts)
            prompt_scores: Dict[int, List[int]] = {i: [] for i in range(len(prompts))}

            for result in results:
                for j, mnemonic in enumerate(result['mnemonics']):
                    prompt_scores[j].append(mnemonic['score'])

            # Вывод результатов
            for result in results:
                st.subheader(f"Слово: {result['word']}")
                for i, mnemonic in enumerate(result['mnemonics']):
                    st.write(f"Промпт {i+1}:")
                    st.write(f"Значение: {mnemonic.get('meaning', 'Не указано')}")
                    st.write(f"Ассоциация: {mnemonic.get('association', 'Не указано')}")
                    st.write(f"Визуальный промпт: {mnemonic.get('prompt', 'Не указано')}")
                    st.write(f"Оценка запоминаемости: {mnemonic.get('score', 'Не оценено')}")
                    st.write("---")

            # Вывод средних оценок запоминаемости для каждого промпта
            st.subheader("Средние оценки запоминаемости для каждого промпта:")
            for i, scores in prompt_scores.items():
                avg_score = mean(scores) if scores else 0
                st.write(f"Промпт {i+1}: {avg_score:.2f}")

            # Опция для сохранения результатов
            if st.button('Сохранить результаты'):
                with open('results.json', 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                st.success('Результаты сохранены в файл results.json')
