import random
from collections import defaultdict
from typing import List, Dict, Tuple, Set
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTableWidget, QTableWidgetItem,
    QPushButton, QVBoxLayout, QWidget, QMessageBox, QFileDialog,
    QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor
import xlwt
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Типы данных
Group = str
Subject = str
Teacher = str
LessonSlot = str
Gene = List  # [Teacher, LessonSlot, Group, Subject]
Schedule = List[Gene]

# Константы
POPULATION_SIZE = 500
GENERATIONS = 1000
ELITISM_RATE = 0.2
SURVIVAL_RATE = 0.8
MAX_LESSONS_PER_DAY = 6
DAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб"]
NUM_THREADS = 4  # Количество потоков для параллельных вычислений

# Временные слоты с реальным временем
TIME_SLOTS = {
    1: "8:30 - 10:00",
    2: "10:10 - 11:40",
    3: "11:50 - 13:20",
    4: "14:05 - 15:35",
    5: "15:55 - 17:25",
    6: "17:35 - 19:05",
    7: "19:15 - 20:45"
}

LESSON_SLOTS = [f"{day}-{n}" for day in DAYS for n in range(1, 8)]  # 7 пар в день для 6 дней

# Тестовые данные
groups = ["ИУ10-11", "ИУ10-12", "ИУ10-13", "ИУ10-14", "ИУ10-15", "ИУ10-16", "ИУ10-17", "ИУ10-18", "ИУ10-19"]
subjects = [
    "интегралы", "языки программирования", "физика", "джава", "физра",
    "Дискретная математика", "Введение в специальность", "ОРГ", "СПО",
    "Основы РБПО", "Матан", "Ангем", "информатика"
]

teachers = [
    "Андрей А", "Кирилл Александрович", "Наталья Андреевна", "Кририлл Дмитриевич", "Валентин Александрович",
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
    ("ИУ10-14", "Информатика"): 2,
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
    "Андрей А": ["СПО", "Информатика"],
    "Кирилл Александрович": ["Языки программирования", "Джава"],
    "Наталья Андреевна": ["Джава"],
    "Кирилл Дмитриевич": ["Языки программирования"],
    "Валлентин Александрович": ["ЯПНУ"],
    "Варвара Александровна": ["Дискретная математика", "Введение в специальность"],
    "Егор Алексеевич": ["Информатика", "Основы РБПО"],
    "Константин Андреевич": ["Языки программирования", "СПО"],
    "Юлия Викторовна": ["Дискретная математика", "ОРГ", ]
}

lecture_groups = {
    "лекция_интегралы": ["ИУ10-11", "ИУ10-12", "ИУ10-15"],
    "лекция_физика": ["ИУ10-11", "ИУ10-13", "ИУ10-16"],
    "лекция_Дискретная математика": ["ИУ10-11", "ИУ10-12", "ИУ10-13"],
    "лекция_информатика": ["ИУ10-12", "ИУ10-14", "ИУ10-19"]
}


class GeneticAlgorithmWorker(QThread):
    finished = pyqtSignal(object)
    progress = pyqtSignal(int)
    message = pyqtSignal(str)

    def __init__(self, blocked_slots):
        super().__init__()
        self.blocked_slots = blocked_slots

    def run(self):
        try:
            schedule = self.genetic_algorithm(self.blocked_slots)
            self.finished.emit(schedule)
        except Exception as e:
            self.message.emit(f"Ошибка: {str(e)}")

    def generate_random_schedule(self, blocked_slots: Set[Tuple[str, int, str]] = None) -> Schedule:
        if blocked_slots is None:
            blocked_slots = set()

        schedule = []

        # Обычные занятия
        for (group, subject), count in group_subject_requirements.items():
            available_teachers = [t for t, subjs in teacher_subjects.items() if subject in subjs]
            if not available_teachers:
                continue

            for _ in range(count):
                available_slots = [
                    slot for slot in LESSON_SLOTS
                    if not is_cell_blocked(
                        slot.split('-')[0],
                        int(slot.split('-')[1]),
                        group,
                        blocked_slots
                    )
                ]
                if not available_slots:
                    continue

                gene = [
                    random.choice(available_teachers),
                    random.choice(available_slots),
                    group,
                    subject
                ]
                schedule.append(gene)

        # Лекции для нескольких групп
        for lecture, groups_list in lecture_groups.items():
            subject = lecture.split('_')[1]
            available_teachers = [t for t, subjs in teacher_subjects.items() if subject in subjs]
            if not available_teachers:
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
                    for group in groups_list
                )
            ]

            if not available_lecture_slots:
                continue

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

        return schedule

    def calculate_fitness(self, schedule: Schedule, blocked_slots: Set[Tuple[str, int, str]] = None) -> int:
        if blocked_slots is None:
            blocked_slots = set()

        if len(schedule) < sum(group_subject_requirements.values()):
            return -1_000_000

        hard_constraints_violations = 0
        teacher_lessons = defaultdict(set)
        group_lessons = defaultdict(set)
        subject_counts = defaultdict(int)
        day_lessons = defaultdict(lambda: defaultdict(int))
        windows_per_group = defaultdict(int)

        for teacher, lesson_slot, group, subject in schedule:
            day, slot_num = lesson_slot.split('-')
            slot_num = int(slot_num)

            if is_cell_blocked(day, slot_num, group, blocked_slots):
                hard_constraints_violations += 10

            if subject.replace("лекция_", "") not in teacher_subjects.get(teacher, []):
                hard_constraints_violations += 1

            if lesson_slot in teacher_lessons[teacher]:
                hard_constraints_violations += 1
            teacher_lessons[teacher].add(lesson_slot)

            if lesson_slot in group_lessons[group]:
                hard_constraints_violations += 1
            group_lessons[group].add(lesson_slot)

            day_lessons[day][group] += 1

            if day_lessons[day][group] > MAX_LESSONS_PER_DAY:
                hard_constraints_violations += 1

            slots = sorted([int(s.split('-')[1]) for s in group_lessons[group] if s.startswith(day)])
            for i in range(1, len(slots)):
                if slots[i] - slots[i - 1] > 1:
                    windows_per_group[group] += 1

            subject_counts[(group, subject)] += 1

        for (group, subject), required in group_subject_requirements.items():
            actual = subject_counts.get((group, subject), 0)
            hard_constraints_violations += abs(required - actual)

        hard_constraints_violations += sum(windows_per_group.values())

        return 1_000_000 - hard_constraints_violations * 10_000

    def calculate_fitness_parallel(self, population, blocked_slots):
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            futures = {executor.submit(self.calculate_fitness, ind, blocked_slots): idx
                       for idx, ind in enumerate(population)}

            results = [0] * len(population)
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

            return results

    def crossover(self, parent1: Schedule, parent2: Schedule) -> Schedule:
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
        if blocked_slots is None:
            blocked_slots = set()

        if not schedule:
            return schedule

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

        population = [self.generate_random_schedule(blocked_slots) for _ in range(POPULATION_SIZE)]

        for generation in range(GENERATIONS):
            # Параллельное вычисление fitness
            fitness_scores = self.calculate_fitness_parallel(population, blocked_slots)
            ranked = sorted(zip(fitness_scores, population), key=lambda x: x[0], reverse=True)

            elite = ranked[:int(POPULATION_SIZE * ELITISM_RATE)]
            survivors = ranked[:int(POPULATION_SIZE * SURVIVAL_RATE)]

            best_fitness, best_schedule = ranked[0]
            self.message.emit(f"Generation {generation}, Best Fitness: {best_fitness}")
            self.progress.emit(int(generation / GENERATIONS * 100))

            if best_fitness >= 900_000:
                self.message.emit(f"Good solution found in generation {generation}")
                return best_schedule

            new_generation = [ind for (fit, ind) in elite]

            while len(new_generation) < POPULATION_SIZE:
                parent1 = self.select_parent(survivors)
                parent2 = self.select_parent(survivors)
                child = self.crossover(parent1, parent2)

                if random.random() < 0.1:
                    child = self.mutate(child, blocked_slots)

                new_generation.append(child)

            population = new_generation

        return max(population, key=lambda x: self.calculate_fitness(x, blocked_slots))


def is_cell_blocked(day: str, time_num: int, group: str, blocked_slots: Set[Tuple[str, int, str]]) -> bool:
    """Проверяет, заблокирована ли ячейка"""
    return (day, time_num, group) in blocked_slots


class ScheduleApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Генератор расписания университета")
        self.setGeometry(100, 100, 1200, 800)
        self.current_schedule = []
        self.blocked_cells = set()
        self.init_ui()

    def init_ui(self):
        self.table = QTableWidget()
        self.table.setColumnCount(len(groups) + 2)
        self.table.setHorizontalHeaderLabels(["День", "Время"] + groups)

        total_rows = len(DAYS) * len(TIME_SLOTS)
        self.table.setRowCount(total_rows)

        row = 0
        for day in DAYS:
            day_item = QTableWidgetItem(day)
            self.table.setItem(row, 0, day_item)
            self.table.setSpan(row, 0, len(TIME_SLOTS), 1)

            for time_num, time_text in TIME_SLOTS.items():
                time_item = QTableWidgetItem(time_text)
                self.table.setItem(row, 1, time_item)

                for col in range(2, self.table.columnCount()):
                    item = QTableWidgetItem("")
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                    self.table.setItem(row, col, item)

                row += 1

        btn_layout = QVBoxLayout()
        self.generate_btn = QPushButton("Сгенерировать расписание")
        self.generate_btn.clicked.connect(self.generate_schedule)
        btn_layout.addWidget(self.generate_btn)

        self.save_btn = QPushButton("Сохранить в Excel")
        self.save_btn.clicked.connect(self.save_to_excel)
        btn_layout.addWidget(self.save_btn)

        self.clear_blocks_btn = QPushButton("Очистить блокировки")
        self.clear_blocks_btn.clicked.connect(self.clear_blocked_cells)
        btn_layout.addWidget(self.clear_blocks_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        btn_layout.addWidget(self.progress_bar)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.table)
        main_layout.addLayout(btn_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

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

    def generate_schedule(self):
        self.generate_btn.setEnabled(False)
        QApplication.processEvents()

        self.worker = GeneticAlgorithmWorker(self.blocked_cells)
        self.worker.finished.connect(self.on_generation_finished)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.message.connect(lambda msg: self.statusBar().showMessage(msg, 5000))
        self.worker.start()

    def on_generation_finished(self, schedule):
        self.current_schedule = schedule
        self.display_schedule()
        self.generate_btn.setEnabled(True)
        QMessageBox.information(self, "Успех", "Расписание успешно сгенерировано!")

    def display_schedule(self):
        schedule_data = {
            day: defaultdict(lambda: defaultdict(str))
            for day in DAYS
        }

        for teacher, lesson_slot, group, subject in self.current_schedule:
            day, time_num = lesson_slot.split('-')
            time_num = int(time_num)
            schedule_data[day][time_num][group] = f"{teacher}: {subject}"

        row = 0
        for day in DAYS:
            for time_num in sorted(TIME_SLOTS.keys()):
                for col, group in enumerate(groups, 2):
                    item_text = schedule_data[day][time_num].get(group, "")
                    item = QTableWidgetItem(item_text)

                    if (day, time_num, group) in self.blocked_cells:
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

            for day in DAYS:
                sheet = workbook.add_sheet(day)
                headers = ["Время"] + groups
                for col, header in enumerate(headers):
                    sheet.write(0, col, header)

                day_data = defaultdict(lambda: defaultdict(str))
                for teacher, lesson_slot, group, subject in self.current_schedule:
                    current_day, time_num = lesson_slot.split('-')
                    if current_day == day:
                        time_num = int(time_num)
                        day_data[time_num][group] = f"{teacher}: {subject}"

                for time_num in sorted(TIME_SLOTS.keys()):
                    row = time_num
                    sheet.write(row, 0, TIME_SLOTS[time_num])
                    for col, group in enumerate(groups, 1):
                        sheet.write(row, col, day_data[time_num].get(group, "---"))

            workbook.save(file_name)
            QMessageBox.information(self, "Успех", f"Расписание сохранено в {file_name}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка сохранения: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ScheduleApp()
    window.show()
    sys.exit(app.exec_())