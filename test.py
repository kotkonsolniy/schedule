import random
from collections import defaultdict
from typing import List, Dict, Tuple, Set
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTableWidget, QTableWidgetItem,
    QPushButton, QVBoxLayout, QWidget, QMessageBox, QFileDialog,
    QProgressBar, QLabel
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QColor, QFont
import xlwt
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

# Типы данных
Group = str
Subject = str
Teacher = str
LessonSlot = str
Gene = List  # [Teacher, LessonSlot, Group, Subject]
Schedule = List[Gene]

# Константы
POPULATION_SIZE = 200
GENERATIONS = 300
ELITISM_RATE = 0.2
SURVIVAL_RATE = 0.8
MAX_LESSONS_PER_DAY = 6
DAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб"]
NUM_THREADS = 4

# Временные слоты
TIME_SLOTS = {
    1: "8:30 - 10:00",
    2: "10:10 - 11:40",
    3: "11:50 - 13:20",
    4: "14:05 - 15:35",
    5: "15:55 - 17:25",
    6: "17:35 - 19:05",
    7: "19:15 - 20:45"
}

LESSON_SLOTS = [f"{day}-{n}" for day in DAYS for n in range(1, 8)]

# Тестовые данные
groups = ["ИУ10-11", "ИУ10-12", "ИУ10-13", "ИУ10-14", "ИУ10-15",
          "ИУ10-16", "ИУ10-17", "ИУ10-18", "ИУ10-19"]

subjects = [
    "интегралы", "языки программирования", "физика", "джава", "физра",
    "Дискретная математика", "Введение в специальность", "ОРГ", "СПО",
    "Основы РБПО", "Матан", "Ангем", "информатика"
]

teachers = [
    "Андрей А", "Кирилл Александрович", "Наталья Андреевна",
    "Кирилл Дмитриевич", "Валентин Александрович",
    "Варвара Александровна", "Егор Алексеевич", "Константин Андреевич",
    "Юлия Викторовна"
]

group_subject_requirements = {
    # ИУ10-11
    ("ИУ10-11", "интегралы"): 4,
    ("ИУ10-11", "физика"): 2,
    ("ИУ10-11", "физра"): 2,
    ("ИУ10-11", "Дискретная математика"): 3,
    ("ИУ10-11", "Матан"): 3,
    ("ИУ10-11", "Ангем"): 2,

    # ИУ10-12
    ("ИУ10-12", "интегралы"): 3,
    ("ИУ10-12", "языки программирования"): 3,
    ("ИУ10-12", "Дискретная математика"): 3,
    ("ИУ10-12", "информатика"): 2,
    ("ИУ10-12", "ОРГ"): 2,

    # ИУ10-13
    ("ИУ10-13", "джава"): 3,
    ("ИУ10-13", "Введение в специальность"): 2,
    ("ИУ10-13", "СПО"): 3,
    ("ИУ10-13", "Основы РБПО"): 2,
    ("ИУ10-13", "физика"): 2,

    # ИУ10-14
    ("ИУ10-14", "языки программирования"): 4,
    ("ИУ10-14", "физра"): 3,
    ("ИУ10-14", "ОРГ"): 3,
    ("ИУ10-14", "информатика"): 2,
    ("ИУ10-14", "Дискретная математика"): 2,

    # ИУ10-15
    ("ИУ10-15", "интегралы"): 4,
    ("ИУ10-15", "СПО"): 3,
    ("ИУ10-15", "Матан"): 3,
    ("ИУ10-15", "Ангем"): 2,
    ("ИУ10-15", "физра"): 2,

    # ИУ10-16
    ("ИУ10-16", "физика"): 3,
    ("ИУ10-16", "Основы РБПО"): 2,
    ("ИУ10-16", "Дискретная математика"): 2,
    ("ИУ10-16", "языки программирования"): 2,

    # ИУ10-17
    ("ИУ10-17", "джава"): 4,
    ("ИУ10-17", "Матан"): 4,
    ("ИУ10-17", "информатика"): 2,
    ("ИУ10-17", "ОРГ"): 2,

    # ИУ10-18
    ("ИУ10-18", "физра"): 2,
    ("ИУ10-18", "Ангем"): 3,
    ("ИУ10-18", "Введение в специальность"): 2,
    ("ИУ10-18", "СПО"): 2,

    # ИУ10-19
    ("ИУ10-19", "языки программирования"): 4,
    ("ИУ10-19", "информатика"): 3,
    ("ИУ10-19", "Дискретная математика"): 2,
    ("ИУ10-19", "Основы РБПО"): 2
}

teacher_subjects = {
    "Андрей А": ["СПО", "информатика"],
    "Кирилл Александрович": ["языки программирования", "джава"],
    "Наталья Андреевна": ["джава"],
    "Кирилл Дмитриевич": ["языки программирования"],
    "Валентин Александрович": ["языки программирования"],
    "Варвара Александровна": ["Дискретная математика", "Введение в специальность"],
    "Егор Алексеевич": ["информатика", "Основы РБПО"],
    "Константин Андреевич": ["языки программирования", "СПО"],
    "Юлия Викторовна": ["Дискретная математика", "ОРГ"]
}

lecture_groups = {
    "лекция_интегралы": ["ИУ10-11", "ИУ10-12", "ИУ10-15"],
    "лекция_физика": ["ИУ10-11", "ИУ10-13", "ИУ10-16"],
    "лекция_Дискретная математика": ["ИУ10-11", "ИУ10-12", "ИУ10-13"],
    "лекция_информатика": ["ИУ10-12", "ИУ10-14", "ИУ10-19"]
}


def is_cell_blocked(day: str, time_num: int, group: str, blocked_slots: Set[Tuple[str, int, str]]) -> bool:
    """Проверяет, заблокирована ли ячейка"""
    return (day, time_num, group) in blocked_slots


class GeneticAlgorithmWorker(QObject):
    finished = pyqtSignal(object)
    progress = pyqtSignal(int)
    message = pyqtSignal(str)
    warning = pyqtSignal(str)

    def __init__(self, blocked_slots):
        super().__init__()
        self.blocked_slots = blocked_slots
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            schedule = self.genetic_algorithm(self.blocked_slots)
            if self._is_running:
                self.finished.emit(schedule)
        except Exception as e:
            if self._is_running:
                self.message.emit(f"Ошибка: {str(e)}\n{traceback.format_exc()}")

    def generate_random_schedule(self, blocked_slots: Set[Tuple[str, int, str]] = None) -> Schedule:
        if blocked_slots is None:
            blocked_slots = set()

        schedule = []
        required_lessons = []

        # Собираем все обязательные занятия
        for (group, subject), count in group_subject_requirements.items():
            for _ in range(count):
                required_lessons.append((group, subject))

        # Перемешиваем для разнообразия
        random.shuffle(required_lessons)

        # Назначаем обязательные занятия
        for group, subject in required_lessons:
            if not self._is_running:
                return []

            available_teachers = [t for t, subjs in teacher_subjects.items()
                                  if subject in subjs]
            if not available_teachers:
                self.warning.emit(f"Нет преподавателя для {subject} в группе {group}")
                continue

            available_slots = [
                slot for slot in LESSON_SLOTS
                if not is_cell_blocked(
                    slot.split('-')[0],
                    int(slot.split('-')[1]),
                    group,
                    blocked_slots
                )
                   and not any(
                    g[1] == slot and g[2] == group
                    for g in schedule
                )
            ]

            if available_slots:
                gene = [
                    random.choice(available_teachers),
                    random.choice(available_slots),
                    group,
                    subject
                ]
                schedule.append(gene)
            else:
                self.warning.emit(f"Нет свободных слотов для {subject} в группе {group}")

        # Добавляем лекции
        for lecture, groups_list in lecture_groups.items():
            if not self._is_running:
                return []

            subject = lecture.split('_')[1]
            available_teachers = [t for t, subjs in teacher_subjects.items()
                                  if subject in subjs]
            if not available_teachers:
                self.warning.emit(f"Нет преподавателя для лекции {subject}")
                continue

            available_lecture_slots = [
                slot for slot in LESSON_SLOTS
                if all(
                    not is_cell_blocked(
                        slot.split('-')[0],
                        int(slot.split('-')[1]),
                        group,
                        blocked_slots
                    )
                    and not any(
                        g[1] == slot and g[2] == group
                        for g in schedule
                    )
                    for group in groups_list
                )
            ]

            if available_lecture_slots:
                slot = random.choice(available_lecture_slots)
                teacher = random.choice(available_teachers)

                for group in groups_list:
                    gene = [
                        teacher,
                        slot,
                        group,
                        f"лекция_{subject}"
                    ]
                    schedule.append(gene)
            else:
                self.warning.emit(f"Нет свободных слотов для лекции {subject}")

        return schedule

    def calculate_fitness(self, schedule: Schedule, blocked_slots: Set[Tuple[str, int, str]] = None) -> int:
        if not self._is_running:
            return 0

        if blocked_slots is None:
            blocked_slots = set()

        # Жесткие ограничения
        hard_penalty = 0

        # 1. Проверка выполнения всех требований
        subject_counts = defaultdict(int)
        for _, _, group, subject in schedule:
            subject_counts[(group, subject)] += 1

        missing_lessons = 0
        for (group, subject), required in group_subject_requirements.items():
            actual = subject_counts.get((group, subject), 0)
            if actual < required:
                missing_lessons += (required - actual) * 1000

        # 2. Проверка конфликтов
        teacher_slots = defaultdict(set)
        group_slots = defaultdict(set)

        for teacher, slot, group, _ in schedule:
            if slot in teacher_slots[teacher]:
                hard_penalty += 100
            teacher_slots[teacher].add(slot)

            if slot in group_slots[group]:
                hard_penalty += 100
            group_slots[group].add(slot)

        # 3. Проверка блокированных слотов
        for teacher, slot, group, _ in schedule:
            day, time_num = slot.split('-')
            if is_cell_blocked(day, int(time_num), group, blocked_slots):
                hard_penalty += 500

        # 4. Проверка максимального количества пар в день
        day_lessons = defaultdict(lambda: defaultdict(int))
        for _, slot, group, _ in schedule:
            day = slot.split('-')[0]
            day_lessons[day][group] += 1
            if day_lessons[day][group] > MAX_LESSONS_PER_DAY:
                hard_penalty += 50

        # Мягкие ограничения (предпочтения)
        soft_penalty = 0

        # 1. Окна между парами
        for group in groups:
            for day in DAYS:
                slots = sorted([
                    int(s.split('-')[1])
                    for s in group_slots[group]
                    if s.startswith(day)
                ])
                for i in range(1, len(slots)):
                    if slots[i] - slots[i - 1] > 1:
                        soft_penalty += 1

        total_penalty = hard_penalty + soft_penalty + missing_lessons
        return 1_000_000 - total_penalty

    def calculate_fitness_parallel(self, population, blocked_slots):
        if not self._is_running:
            return [0] * len(population)

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            futures = {executor.submit(self.calculate_fitness, ind, blocked_slots): idx
                       for idx, ind in enumerate(population)}

            results = [0] * len(population)
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
                if not self._is_running:
                    return [0] * len(population)

            return results

    def crossover(self, parent1: Schedule, parent2: Schedule) -> Schedule:
        if not self._is_running:
            return []

        child = []
        parent2_genes = parent2.copy()

        for gene in parent1:
            teacher, lesson_slot, group, subject = gene

            teacher_conflict = sum(1 for t, ls, g, s in parent1
                                   if t == teacher and ls == lesson_slot) != 1
            group_conflict = sum(1 for t, ls, g, s in parent1
                                 if g == group and ls == lesson_slot) != 1

            if not teacher_conflict and not group_conflict:
                child.append(gene)
            else:
                alternatives = [g for g in parent2_genes
                                if g[2] == group and g[3] == subject]

                valid_alternatives = []
                for alt in alternatives:
                    alt_teacher, alt_lesson, alt_group, alt_subject = alt
                    teacher_ok = sum(1 for t, l, g, s in child
                                     if t == alt_teacher and l == alt_lesson) == 0
                    group_ok = sum(1 for t, l, g, s in child
                                   if g == alt_group and l == alt_lesson) == 0
                    if teacher_ok and group_ok:
                        valid_alternatives.append(alt)

                if valid_alternatives:
                    chosen = random.choice(valid_alternatives)
                    child.append(chosen)
                    if chosen in parent2_genes:
                        parent2_genes.remove(chosen)
                elif alternatives:
                    child.append(random.choice(alternatives))
                else:
                    child.append(gene)

        return child

    def mutate(self, schedule: Schedule, blocked_slots: Set[Tuple[str, int, str]] = None) -> Schedule:
        if not self._is_running or not schedule:
            return schedule

        if blocked_slots is None:
            blocked_slots = set()

        idx = random.randint(0, len(schedule) - 1)
        teacher, lesson_slot, group, subject = schedule[idx]

        teacher_conflict = sum(1 for t, ls, g, s in schedule
                               if t == teacher and ls == lesson_slot) != 1
        group_conflict = sum(1 for t, ls, g, s in schedule
                             if g == group and ls == lesson_slot) != 1

        if not teacher_conflict and not group_conflict and random.random() < 0.7:
            return schedule

        mutation_type = random.random()

        if mutation_type < 0.4:
            available_teachers = [t for t, subjs in teacher_subjects.items()
                                  if subject.replace("лекция_", "") in subjs]
            if available_teachers:
                schedule[idx][0] = random.choice(available_teachers)

        elif mutation_type < 0.8:
            current_day = lesson_slot.split('-')[0]
            available_slots = [ls for ls in LESSON_SLOTS
                               if ls.split('-')[0] == current_day and
                               not any(g[2] == group and g[1] == ls
                                       for g in schedule if g != schedule[idx]) and
                               not is_cell_blocked(
                                   ls.split('-')[0],
                                   int(ls.split('-')[1]),
                                   group,
                                   blocked_slots
                               )]
            if available_slots:
                schedule[idx][1] = random.choice(available_slots)
            else:
                available_slots = [ls for ls in LESSON_SLOTS
                                   if not any(g[2] == group and g[1] == ls
                                              for g in schedule if g != schedule[idx]) and
                                   not is_cell_blocked(
                                       ls.split('-')[0],
                                       int(ls.split('-')[1]),
                                       group,
                                       blocked_slots
                                   )]
                if available_slots:
                    schedule[idx][1] = random.choice(available_slots)

        return schedule

    def select_parent(self, ranked_population: List[Tuple[int, Schedule]]) -> Schedule:
        if not self._is_running or not ranked_population:
            return []

        total_fitness = sum(max(fit, 0) for fit, ind in ranked_population)
        if total_fitness == 0:
            return random.choice(ranked_population)[1]

        pick = random.uniform(0, total_fitness)
        current = 0
        for fit, ind in ranked_population:
            current += max(fit, 0)
            if current > pick:
                return ind
        return ranked_population[0][1]

    def genetic_algorithm(self, blocked_slots: Set[Tuple[str, int, str]] = None) -> Schedule:
        if blocked_slots is None:
            blocked_slots = set()

        population = []
        for _ in range(POPULATION_SIZE):
            if not self._is_running:
                return []
            population.append(self.generate_random_schedule(blocked_slots))

        for generation in range(GENERATIONS):
            if not self._is_running:
                return []

            fitness_scores = self.calculate_fitness_parallel(population, blocked_slots)
            ranked = sorted(zip(fitness_scores, population), key=lambda x: x[0], reverse=True)

            elite = ranked[:int(POPULATION_SIZE * ELITISM_RATE)]
            survivors = ranked[:int(POPULATION_SIZE * SURVIVAL_RATE)]

            best_fitness, best_schedule = ranked[0]
            self.message.emit(f"Поколение {generation}, Лучшая пригодность: {best_fitness}")
            self.progress.emit(int(generation / GENERATIONS * 100))

            if best_fitness >= 900_000:
                self.message.emit(f"Хорошее решение найдено в поколении {generation}")
                return best_schedule

            new_generation = [ind for (fit, ind) in elite]

            while len(new_generation) < POPULATION_SIZE:
                if not self._is_running:
                    return []

                parent1 = self.select_parent(survivors)
                parent2 = self.select_parent(survivors)
                child = self.crossover(parent1, parent2)

                if random.random() < 0.1:
                    child = self.mutate(child, blocked_slots)

                new_generation.append(child)

            population = new_generation

        return max(population, key=lambda x: self.calculate_fitness(x, blocked_slots))


class ScheduleApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Генератор расписания университета")
        self.setGeometry(100, 100, 1200, 900)
        self.current_schedule = []
        self.blocked_cells = set()
        self.worker_thread = None
        self.worker = None
        self.init_ui()

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Таблица расписания
        self.table = QTableWidget()
        self.table.setColumnCount(len(groups) + 2)
        self.table.setHorizontalHeaderLabels(["День", "Время"] + groups)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setFont(QFont("Arial", 10))

        # Настройка размеров столбцов
        self.table.setColumnWidth(0, 80)
        self.table.setColumnWidth(1, 150)
        for i in range(2, len(groups) + 2):
            self.table.setColumnWidth(i, 180)

        total_rows = len(DAYS) * len(TIME_SLOTS)
        self.table.setRowCount(total_rows)

        # Заполняем столбцы дней и времени
        row = 0
        for day in DAYS:
            day_item = QTableWidgetItem(day)
            day_item.setTextAlignment(Qt.AlignCenter)
            day_item.setFont(QFont("Arial", 10, QFont.Bold))
            self.table.setItem(row, 0, day_item)
            self.table.setSpan(row, 0, len(TIME_SLOTS), 1)

            for time_num, time_text in TIME_SLOTS.items():
                time_item = QTableWidgetItem(time_text)
                time_item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row, 1, time_item)

                for col in range(2, self.table.columnCount()):
                    item = QTableWidgetItem("")
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.table.setItem(row, col, item)

                row += 1

        # Панель управления
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)

        # Кнопки управления
        self.generate_btn = QPushButton("Сгенерировать расписание")
        self.generate_btn.setFont(QFont("Arial", 10))
        self.generate_btn.clicked.connect(self.generate_schedule)

        self.save_btn = QPushButton("Сохранить в Excel")
        self.save_btn.setFont(QFont("Arial", 10))
        self.save_btn.clicked.connect(self.save_to_excel)

        self.clear_blocks_btn = QPushButton("Очистить блокировки")
        self.clear_blocks_btn.setFont(QFont("Arial", 10))
        self.clear_blocks_btn.clicked.connect(self.clear_blocked_cells)

        # Прогресс-бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)

        # Статусная строка
        self.status_label = QLabel("Готово к работе")
        self.status_label.setFont(QFont("Arial", 9))
        self.status_label.setAlignment(Qt.AlignCenter)

        # Добавляем элементы на панель
        control_layout.addWidget(self.generate_btn)
        control_layout.addWidget(self.save_btn)
        control_layout.addWidget(self.clear_blocks_btn)
        control_layout.addWidget(self.progress_bar)
        control_layout.addWidget(self.status_label)

        # Добавляем таблицу и панель управления в основной layout
        self.layout.addWidget(self.table)
        self.layout.addWidget(control_panel)

        # Подключаем обработчик кликов по ячейкам
        self.table.cellClicked.connect(self.toggle_cell_block)

    def toggle_cell_block(self, row, column):
        if column < 2:
            return

        day, time_num = self.get_day_and_time_from_row(row)
        if day is None:
            return

        group = groups[column - 2]
        cell_key = (day, time_num, group)

        if cell_key in self.blocked_cells:
            self.blocked_cells.remove(cell_key)
            self.table.item(row, column).setBackground(Qt.white)
        else:
            self.blocked_cells.add(cell_key)
            self.table.item(row, column).setBackground(QColor(220, 220, 220))

    def get_day_and_time_from_row(self, row):
        day_row = 0
        current_day = None
        for day in DAYS:
            if row < day_row + len(TIME_SLOTS):
                current_day = day
                break
            day_row += len(TIME_SLOTS)
        else:
            return None, None

        time_num = (row - day_row) % len(TIME_SLOTS) + 1
        return current_day, time_num

    def clear_blocked_cells(self):
        self.blocked_cells.clear()
        for row in range(self.table.rowCount()):
            for col in range(2, self.table.columnCount()):
                if self.table.item(row, col):
                    self.table.item(row, col).setBackground(Qt.white)
        self.status_label.setText("Все блокировки очищены")

    def generate_schedule(self):
        if self.worker_thread is not None and self.worker_thread.isRunning():
            self.worker.stop()
            self.worker_thread.quit()
            self.worker_thread.wait()

        self.generate_btn.setEnabled(False)
        self.status_label.setText("Генерация расписания...")
        QApplication.processEvents()

        self.worker_thread = QThread()
        self.worker = GeneticAlgorithmWorker(self.blocked_cells)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_generation_finished)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.message.connect(self.status_label.setText)
        self.worker.warning.connect(self.status_label.setText)

        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.start()

    def on_generation_finished(self, schedule):
        self.current_schedule = schedule
        self.display_schedule()
        self.generate_btn.setEnabled(True)

        # Проверяем, все ли занятия назначены
        missing = self.check_missing_lessons()
        if missing:
            self.status_label.setText(f"Пропущено занятий: {missing}")
            QMessageBox.warning(self, "Внимание",
                                f"Не все занятия назначены!\nПропущено: {missing}")
        else:
            self.status_label.setText("Все занятия успешно назначены!")
            QMessageBox.information(self, "Успех", "Расписание успешно сгенерировано!")

    def check_missing_lessons(self):
        subject_counts = defaultdict(int)
        for _, _, group, subject in self.current_schedule:
            subject_counts[(group, subject)] += 1

        missing = []
        for (group, subject), required in group_subject_requirements.items():
            actual = subject_counts.get((group, subject), 0)
            if actual < required:
                missing.append(f"{group}: {subject} ({actual}/{required})")

        return ", ".join(missing) if missing else None

    def display_schedule(self):
        # Словарь для хранения всех занятий
        schedule_data = defaultdict(lambda: defaultdict(list))

        for teacher, lesson_slot, group, subject in self.current_schedule:
            day, time_num = lesson_slot.split('-')
            time_num = int(time_num)
            schedule_data[(day, time_num, group)].append(f"{teacher}: {subject}")

        # Отображаем все занятия
        row = 0
        for day in DAYS:
            for time_num in sorted(TIME_SLOTS.keys()):
                for col, group in enumerate(groups, 2):
                    key = (day, time_num, group)
                    lessons = schedule_data[key]
                    item_text = "\n".join(lessons) if lessons else ""

                    item = QTableWidgetItem(item_text)
                    item.setFont(QFont("Arial", 9))

                    if key in self.blocked_cells:
                        item.setBackground(QColor(220, 220, 220))

                    self.table.setItem(row, col, item)
                row += 1

    def save_to_excel(self):
        if not self.current_schedule:
            QMessageBox.warning(self, "Ошибка", "Нет данных для сохранения")
            return

        try:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Сохранить файл Excel", "", "Excel Files (*.xls)", options=options
            )
            if not file_name:
                return

            workbook = xlwt.Workbook()
            style = xlwt.easyxf('align: vert centre, horiz center')

            for day in DAYS:
                sheet = workbook.add_sheet(day)

                # Заголовки
                headers = ["Время"] + groups
                for col, header in enumerate(headers):
                    sheet.write(0, col, header, style)
                    sheet.col(col).width = 4000  # Ширина столбцов

                # Данные
                day_data = defaultdict(lambda: defaultdict(list))
                for teacher, lesson_slot, group, subject in self.current_schedule:
                    current_day, time_num = lesson_slot.split('-')
                    if current_day == day:
                        time_num = int(time_num)
                        day_data[time_num][group].append(f"{teacher}: {subject}")

                for time_num in sorted(TIME_SLOTS.keys()):
                    row = time_num
                    sheet.write(row, 0, TIME_SLOTS[time_num], style)

                    for col, group in enumerate(groups, 1):
                        lessons = day_data[time_num].get(group, [])
                        sheet.write(row, col, "\n".join(lessons), style)

            workbook.save(file_name)
            self.status_label.setText(f"Файл сохранен: {file_name}")
            QMessageBox.information(self, "Успех", f"Расписание сохранено в {file_name}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка сохранения: {str(e)}")
            self.status_label.setText(f"Ошибка сохранения: {str(e)}")

    def closeEvent(self, event):
        if self.worker_thread is not None and self.worker_thread.isRunning():
            self.worker.stop()
            self.worker_thread.quit()
            self.worker_thread.wait()
        event.accept()


if __name__ == "__main__":
    try:
        # Увеличиваем размер стека для основного потока (Windows)
        if sys.platform == "win32":
            import ctypes

            ctypes.windll.kernel32.SetThreadStackGuarantee(ctypes.c_ulong(0x10000))

        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        window = ScheduleApp()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Critical error: {str(e)}\n{traceback.format_exc()}")
        sys.exit(1)