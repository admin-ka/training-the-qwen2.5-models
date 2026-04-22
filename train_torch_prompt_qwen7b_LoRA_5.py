#!/usr/bin/env python3
"""
Обучение Qwen2.5 с LoRA на CPU
"""

print("=== СКРИПТ ЗАПУЩЕН ===", flush=True)
import sys
sys.stdout.flush()

import os
print("os импортирован", flush=True)

import json
print("json импортирован", flush=True)

import torch
print(f"torch импортирован: {torch.__version__}", flush=True)

# Дальше остальные импорты...

# ===== НАСТРОЙКА ПОТОКОВ И ПАМЯТИ =====
os.environ['OMP_NUM_THREADS'] = '24'
os.environ['MKL_NUM_THREADS'] = '24'
os.environ['OPENBLAS_NUM_THREADS'] = '24'
os.environ['VECLIB_MAXIMUM_THREADS'] = '24'
os.environ['NUMEXPR_NUM_THREADS'] = '24'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

torch.set_num_threads(24)
torch.set_num_interop_threads(6)
torch.backends.cudnn.enabled = False

from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import torch.optim as optim
from tqdm import tqdm

# ===== ПУТИ =====
DATASET_PATH = os.path.expanduser("~/Documents/My_AI/dataset.jsonl")
MODEL_PATH = os.path.expanduser("~/Builds/Qwen/Qwen2.5-7B")
OUTPUT_DIR = os.path.expanduser("~/Documents/My_AI/qwen_lora_finetuned")

# ===== ПАРАМЕТРЫ ОБУЧЕНИЯ =====
BATCH_SIZE = 1
EPOCHS = 1
LEARNING_RATE = 5e-4
MAX_LENGTH = 256 #MAX_LENGTH = 384
ACCUMULATION_STEPS = 2 #ACCUMULATION_STEPS = 8
GRADIENT_CLIP = 1.0
SAVE_EVERY_STEPS = 50 #SAVE_EVERY_STEPS = 25

# ===== КЛАСС ДАТАСЕТА =====

class JSONLDataset(Dataset):
    """Датасет для загрузки JSONL файлов с chat-формате (messages)"""
    
    def __init__(self, file_path, tokenizer, max_length=384):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        print(f"Загружаю датасет из {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        self.data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Ошибка парсинга строки {idx + 1}: {e}")
                        continue
        
        if not self.data:
            raise ValueError("❌ Датасет пуст или не загружен")
        
        print(f"✓ Загружено {len(self.data)} примеров\n")
    
    def _format_messages(self, messages):
        """
        Преобразование массива messages в текст для обучения.
        Формат: <|im_start|>user\nПромпт<|im_end|>\n<|im_start|>assistant\nОтвет<|im_end|>
        """
        formatted_text = ""
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            # Используем формат Qwen2.5
            if role == 'user':
                formatted_text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == 'assistant':
                formatted_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            else:
                formatted_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        
        return formatted_text.strip()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Получаем массив messages
        messages = item.get('messages', [])
        
        if not messages:
            print(f"⚠️  Пустой массив messages в индексе {idx}, пропускаю")
            return self.__getitem__(np.random.randint(0, len(self.data)))
        
        # Форматируем сообщения в текст
        text = self._format_messages(messages)
        
        if not text or not text.strip():
            print(f"⚠️  Пустой текст в индексе {idx}, пропускаю")
            return self.__getitem__(np.random.randint(0, len(self.data)))
        
        # Токенизируем
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encodings['input_ids'].squeeze(0)
        attention_mask = encodings['attention_mask'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }


# ===== ЗАГРУЗКА МОДЕЛИ С LoRA =====
def load_model_with_lora(model_path):
    """Загрузка модели и применение LoRA конфигурации"""
    
    print(f"Загружаю модель из {model_path}...\n")
    
    # ===== ЗАГРУЗКА TOKENIZER =====
    print("Загружаю tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )
        print("✓ Tokenizer загружен успешно (slow tokenizer)")
        
    except Exception as e:
        print(f"Ошибка при загрузке tokenizer: {e}")
        print("Пытаюсь загрузить tokenizer.json вручную...\n")
        
        try:
            from tokenizers import Tokenizer
            from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
            
            tokenizer_json_path = os.path.join(model_path, "tokenizer.json")
            
            if os.path.exists(tokenizer_json_path):
                tokenizers_obj = Tokenizer.from_file(tokenizer_json_path)
                
                config_path = os.path.join(model_path, "tokenizer_config.json")
                if os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                else:
                    config = {}
                
                tokenizer = PreTrainedTokenizerFast(
                    tokenizer_object=tokenizers_obj,
                    model_max_length=2048,
                    **config
                )
                print("✓ Tokenizer загружен из tokenizer.json")
            else:
                print(f"✗ tokenizer.json не найден в {model_path}")
                raise FileNotFoundError(f"Нет tokenizer файлов в {model_path}")
                
        except Exception as e2:
            print(f"✗ Не удалось загрузить tokenizer: {e2}")
            raise
    
    # Установите pad_token если его нет
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # ===== ЗАГРУЗКА МОДЕЛИ =====
    print("\nЗагружаю саму модель...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # ← ИЗМЕНЕНО: float32 → float16
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    print("✓ Модель загружена успешно (float16)")
    
    # ===== КОНФИГУРАЦИЯ LoRA =====
    print("Применяю LoRA конфигурацию...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        modules_to_save=["lm_head"],
    )
    
    model = get_peft_model(model, lora_config)
    print("✓ LoRA применена успешно\n")
    
    # Вывод статистики
    model.print_trainable_parameters()
    print()
    
    return model, tokenizer


# ===== ОБУЧЕНИЕ =====
def train(
    model,
    tokenizer,
    train_loader,
    epochs=1,
    learning_rate=5e-4,
    accumulation_steps=8,
    output_dir="./qwen_lora_finetuned",
    gradient_clip=1.0,
    save_every_steps=25
):
    """Основной цикл обучения на CPU"""
    
    import time
    import sys
    
    device = torch.device("cpu")
    model.to(device)
    model.train()
    
    # ===== ИЗМЕНЕНО: Установите float16 как тип данных по умолчанию =====
    #torch.set_default_dtype(torch.float32)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("НАЧИНАЮ ОБУЧЕНИЕ")
    print("="*60 + "\n")
    
    total_steps = 0
    epoch_start_time = time.time()
    
    for epoch in range(epochs):
        print(f"{'='*60}")
        print(f"Эпоха {epoch + 1}/{epochs}")
        print(f"{'='*60}\n")
        
        epoch_loss = 0
        step_in_epoch = 0
        epoch_start = time.time()
        
        try:
            progress_bar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Обучение эпоха {epoch + 1}"
            )
            
            for batch_idx, batch in progress_bar:
                step_start = time.time()
                
                # Подготовка данных
                input_ids = batch['input_ids'].to(device, dtype=torch.long)
                attention_mask = batch['attention_mask'].to(device, dtype=torch.long)
                labels = batch['labels'].to(device, dtype=torch.long)
                
                # FORWARD PASS
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                loss = loss / accumulation_steps
                
                # BACKWARD PASS
                loss.backward()
                
                # OPTIMIZER STEP (каждые accumulation_steps шагов)
                if (batch_idx + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=gradient_clip
                    )
                    optimizer.step()
                    optimizer.zero_grad()
                    total_steps += 1
                
                step_time = time.time() - step_start
                epoch_loss += loss.item()
                step_in_epoch += 1
                
                # ЛОГИРУЙТЕ КАЖДЫЙ ШАГ
                avg_loss = epoch_loss / step_in_epoch
                
                progress_bar.set_postfix({
                    'loss': f"{loss.item() * accumulation_steps:.4f}",
                    'avg_loss': f"{avg_loss:.4f}",
                    'step': total_steps,
                    'time': f"{step_time:.2f}s"
                })
                
                # Очистка памяти
                del input_ids, attention_mask, labels, outputs
                torch.cuda.empty_cache()
                
                # СОХРАНЯЙТЕ ЧЕКПОИНТЫ
                if total_steps > 0 and total_steps % save_every_steps == 0:
                    checkpoint_dir = f"{output_dir}/checkpoint-{total_steps}"
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    print(f"\n💾 Сохраняю чекпоинт на шаге {total_steps}: {checkpoint_dir}")
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    print(f"✓ Чекпоинт сохранён\n")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Обучение прервано пользователем")
            break
        except Exception as e:
            print(f"\n\n❌ Ошибка при обучении: {e}")
            import traceback
            traceback.print_exc()
            break
        
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / max(step_in_epoch, 1)
        
        print(f"\n{'='*60}")
        print(f"✓ Эпоха {epoch + 1} завершена за {epoch_time:.1f}s")
        print(f"  Средняя потеря: {avg_epoch_loss:.4f}")
        print(f"  Всего шагов: {total_steps}")
        print(f"{'='*60}\n")
        
        # Сохранение модели после эпохи
        epoch_save_dir = f"{output_dir}/epoch-{epoch + 1}"
        os.makedirs(epoch_save_dir, exist_ok=True)
        print(f"📁 Сохраняю модель эпохи: {epoch_save_dir}")
        model.save_pretrained(epoch_save_dir)
        tokenizer.save_pretrained(epoch_save_dir)
        print("✓ Модель сохранена\n")
    
    total_time = time.time() - epoch_start_time
    
    print("\n" + "="*60)
    print("✓ ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"  Общее время: {total_time:.1f}s")
    print(f"  Всего шагов: {total_steps}")
    print("="*60)
    
    # Сохранение финальной модели
    print(f"\n💾 Сохраняю финальную модель в {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✓ Модель сохранена в {output_dir}\n")
    
    return model


# ===== ФУНКЦИЯ ДЛЯ ИСПОЛЬЗОВАНИЯ ОБУЧЕННОЙ МОДЕЛИ =====
def load_finetuned_model(lora_path):
    """Загрузка обученной модели с LoRA"""
    from peft import AutoPeftModelForCausalLM
    
    print(f"Загружаю обученную модель из {lora_path}...")
    model = AutoPeftModelForCausalLM.from_pretrained(
        lora_path,
        torch_dtype=torch.float16,  # ← ИЗМЕНЕНО: float16
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        lora_path,
        trust_remote_code=True,
        use_fast=False
    )
    print("✓ Модель загружена\n")
    return model, tokenizer


# ===== ФУНКЦИЯ ДЛЯ ГЕНЕРАЦИИ ТЕКСТА =====
def generate_text(model, tokenizer, prompt, max_length=200):
    """Генерация текста с использованием обученной модели"""
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=torch.ones_like(input_ids)
        )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


# ===== ГЛАВНАЯ ФУНКЦИЯ =====
def main():
    """Запуск обучения"""
    
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ МОДЕЛИ QWEN2.5 С LoRA (float16)")
    print("=" * 60)
    print(f"\n📊 Конфигурация:")
    print(f"  Потоков CPU: 24")
    print(f"  Памяти ОЗУ: 32 Гб")
    print(f"  Датасет: {DATASET_PATH}")
    print(f"  Модель: {MODEL_PATH}")
    print(f"  Выход: {OUTPUT_DIR}")
    print(f"  Тип данных: float16")
    
    print(f"\n⚙️  Параметры обучения:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Эпох: {EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max length: {MAX_LENGTH}")
    print(f"  Accumulation steps: {ACCUMULATION_STEPS}")
    print(f"  Сохранение каждые: {SAVE_EVERY_STEPS} шагов")
    print()
    try:
        # Загрузка модели
        model, tokenizer = load_model_with_lora(MODEL_PATH)
        
        # Загрузка датасета
        dataset = JSONLDataset(DATASET_PATH, tokenizer, max_length=MAX_LENGTH)
        
        # Создание DataLoader
        train_loader = DataLoader(
            dataset,
            batch_size=1, #batch_size=BATCH_SIZE,
            shuffle=False, #shuffle=True,
            num_workers=4, #num_workers=12,
            pin_memory=False, #pin_memory=True,
	    persistent_workers=True, #persistent_workers=True,
	    prefetch_factor=2 #prefetch_factor=2
        )
        
        print(f"📦 Батчей на эпоху: {len(train_loader)}\n")
        
        # Обучение
        model = train(
            model=model,
            tokenizer=tokenizer,
            train_loader=train_loader,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            accumulation_steps=ACCUMULATION_STEPS,
            output_dir=OUTPUT_DIR,
            gradient_clip=GRADIENT_CLIP,
            save_every_steps=SAVE_EVERY_STEPS
        )
        
        print("✅ Процесс завершен успешно!")
        print(f"\n💾 Для загрузки модели используйте:")
        print(f"   from peft import AutoPeftModelForCausalLM")
        print(f"   model = AutoPeftModelForCausalLM.from_pretrained('{OUTPUT_DIR}')\n")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":    main()
