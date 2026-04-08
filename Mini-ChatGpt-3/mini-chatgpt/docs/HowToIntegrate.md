# Інструкція з інтеграції (для Етапу 3)



## 1. Архітектура конвеєра

Клас `RuntimeTextGenerator` працює так:

1. Отримання тексту від користувача.

2. Перетворення тексту в ID токенів (через `ITokenizer`).

3. Отримання оцінок від мережі (через `ILanguageModel.NextTokenScores`).

4. Нормалізація оцінок у ймовірності (через `IMathOps.Softmax`).

5. Вибір наступного токена (через `ISampler` з урахуванням `Temp` та `Top-K`).

6. Повернення згенерованого тексту в `ChatRepl`.



## 2. Запуск застосунку із вашими готовими частинами

Потрібно зібрати всі компоненти в одній точці входу (як варіант у `Program.cs` головного проєкту):



```csharp

using MiniChatGPT.Contracts;

using Lib.MathCore;

using Lib.Runtime;

using MiniChatGPT.ChatConsole;

using MiniChatGPT.Sampling;



IMathOps mathOps = new MathOps(); 

ILanguageModel model = new YoursModel(); 

ITokenizer tokenizer = new YoursTokenizer(); 

ISampler sampler = new Sampler(mathOps);



ITextGenerator generator = new RuntimeTextGenerator(model, sampler, mathOps);



var chat = new ChatRepl(generator);

chat.Run(temp: 0.5f, topK: 10, seed: null);

