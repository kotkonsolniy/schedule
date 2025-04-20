import random
from collections import defaultdict
from typing import List, Dict, Tuple, Set
from tabulate import tabulate
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTableWidget,
                             QTableWidgetItem, QPushButton, QVBoxLayout,
                             QWidget, QLabel, QComboBox, QMessageBox)
from PyQt5.QtCore import Qt
import json
import sys

# Типы данных
Group = str
Subject = str
Teacher = str
LessonSlot = str
Gene = List  # [Teacher, LessonSlot, Group, Subject, WeekType]
Schedule = List[Gene]
WeekType = str  # "числитель" или "знаменатель"

# Константы
POPULATION_SIZE = 500
GENERATIONS = 1000
ELITISM_RATE = 0.2
SURVIVAL_RATE = 0.8
MAX_LESSONS_PER_DAY = 6
DAYS = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб"]
LESSON_SLOTS = [f"{day}-{n}" for day in DAYS for n in range(1, 7)]
WEEK_TYPES = ["числитель", "знаменатель"]

# Тестовые данные
groups = ["ИУ10-11", "ИУ10-12", "ИУ10-13", "ИУ10-14", "ИУ10-15"]
subjects = ["интегралы", "япы", "физика", "джава", "физра"]
teachers = ["Иван", "Кирилл", "Варя", "Оля", "Коля"]

group_subject_requirements = {
    ("ИУ10-11", "интегралы"): 3,
    ("ИУ10-11", "физика"): 1,
    ("ИУ10-12", "интегралы"): 1,
    ("ИУ10-12", "япы"): 1,
    ("ИУ10-13", "джава"): 2,
    ("ИУ10-14", "япы"): 2,
    ("ИУ10-14", "физра"): 3,
    ("ИУ10-15", "интегралы"): 3
}

teacher_subjects = {
    "Иван": ["интегралы", "физика"],
    "Кирилл": ["джава"],
    "Варя": ["физра"],
    "Коля": ["япы"],
    "Оля": ["япы", "джаva"]
}

lecture_groups = {  # Лекции для нескольких групп
    "лекция_интегралы": ["ИУ10-11", "ИУ10-12", "ИУ10-15"],
    "лекция_физика": ["ИУ10-11", "ИУ10-13"]
}


def generate_random_schedule() -> Schedule:
    schedule = []

    # Обычные занятия
    for (group, subject), count in group_subject_requirements.items():
        available_teachers = [t for t, subjs in teacher_subjects.items() if subject in subjs]
        if not available_teachers:
            continue

        for _ in range(count):
            week_type = random.choice(WEEK_TYPES)
            gene = [
                random.choice(available_teachers),
                random.choice(LESSON_SLOTS),
                group,
                subject,
                week_type
            ]
            schedule.append(gene)

    # Лекции для нескольких групп
    for lecture, groups_list in lecture_groups.items():
        subject = lecture.split('_')[1]
        available_teachers = [t for t, subjs in teacher_subjects.items() if subject in subjs]
        if not available_teachers:
            continue

        week_type = random.choice(WEEK_TYPES)
        slot = random.choice(LESSON_SLOTS)
        teacher = random.choice(available_teachers)

        for group in groups_list:
            gene = [
                teacher,
                slot,
                group,
                f"лекция_{subject}",
                week_type
            ]
            schedule.append(gene)

    return schedule


def calculate_fitness(schedule: Schedule) -> int:
    if len(schedule) < sum(group_subject_requirements.values()):
        return -1_000_000

    hard_constraints_violations = 0
    teacher_lessons = defaultdict(set)
    group_lessons = defaultdict(lambda: defaultdict(set))
    subject_counts = defaultdict(int)
    day_lessons = defaultdict(lambda: defaultdict(int))
    windows_per_group = defaultdict(int)
    free_days_per_group = defaultdict(set)

    for teacher, lesson_slot, group, subject, week_type in schedule:
        day, slot_num = lesson_slot.split('-')
        slot_num = int(slot_num)

        # Проверка преподавателя
        if subject.replace("лекция_", "") not in teacher_subjects.get(teacher, []):
            hard_constraints_violations += 1

        # Проверка двойного бронирования преподавателя
        teacher_key = (teacher, week_type)
        if lesson_slot in teacher_lessons[teacher_key]:
            hard_constraints_violations += 1
        teacher_lessons[teacher_key].add(lesson_slot)

        # Проверка двойного бронирования группы
        group_key = (group, week_type)
        if lesson_slot in group_lessons[group_key]['slots']:
            hard_constraints_violations += 1
        group_lessons[group_key]['slots'].add(lesson_slot)

        # Подсчет пар в день
        day_key = (day, week_type)
        day_lessons[day_key][group] += 1

        # Проверка на превышение лимита пар
        if day_lessons[day_key][group] > MAX_LESSONS_PER_DAY:
            hard_constraints_violations += 1

        # Подсчет окон
        slots = sorted([int(s.split('-')[1]) for s in group_lessons[group_key]['slots'] if s.startswith(day)])
        for i in range(1, len(slots)):
            if slots[i] - slots[i - 1] > 1:
                windows_per_group[group] += 1

        # Подсчет выходных
        if day_lessons[day_key][group] > 0:
            free_days_per_group[group_key].add(day)

        subject_counts[(group, subject, week_type)] += 1

    # Проверка требований по предметам
    for (group, subject), required in group_subject_requirements.items():
        actual = subject_counts.get((group, subject, "числитель"), 0) + \
                 subject_counts.get((group, subject, "знаменатель"), 0)
        hard_constraints_violations += abs(required - actual)

    # Штраф за окна
    hard_constraints_violations += sum(windows_per_group.values())

    # Поощрение за оптимальные выходные (минимум 1 полный выходной)
    for group in groups:
        for week_type in WEEK_TYPES:
            key = (group, week_type)
            if len(free_days_per_group.get(key, set())) >= len(DAYS) - 1:
                hard_constraints_violations += 5  # Штраф за отсутствие выходных

    return 1_000_000 - hard_constraints_violations * 10_000


def crossover(parent1: Schedule, parent2: Schedule) -> Schedule:
    child = []
    parent2_genes = parent2.copy()

    for gene in parent1:
        teacher, lesson_slot, group, subject, week_type = gene

        # Проверка конфликтов
        teacher_conflict = sum(1 for t, ls, g, s, wt in parent1
                               if t == teacher and ls == lesson_slot and wt == week_type) != 1
        group_conflict = sum(1 for t, ls, g, s, wt in parent1
                             if g == group and ls == lesson_slot and wt == week_type) != 1

        if not teacher_conflict and not group_conflict:
            child.append(gene)
        else:
            # Поиск альтернатив во втором родителе
            alternatives = [g for g in parent2_genes
                            if g[2] == group and g[3] == subject and g[4] == week_type]

            # Фильтр по конфликтам
            valid_alternatives = []
            for alt in alternatives:
                alt_teacher, alt_lesson, alt_group, alt_subject, alt_week = alt
                teacher_ok = sum(1 for t, l, g, s, w in child
                                 if t == alt_teacher and l == alt_lesson and w == alt_week) == 0
                group_ok = sum(1 for t, l, g, s, w in child
                               if g == alt_group and l == alt_lesson and w == alt_week) == 0
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


def mutate(schedule: Schedule) -> Schedule:
    if not schedule:
        return schedule

    idx = random.randint(0, len(schedule) - 1)
    teacher, lesson_slot, group, subject, week_type = schedule[idx]

    # Проверка текущего гена на конфликты
    teacher_conflict = sum(1 for t, ls, g, s, wt in schedule
                           if t == teacher and ls == lesson_slot and wt == week_type) != 1
    group_conflict = sum(1 for t, ls, g, s, wt in schedule
                         if g == group and ls == lesson_slot and wt == week_type) != 1

    if not teacher_conflict and not group_conflict and random.random() < 0.7:
        return schedule  # 70% шанс не мутировать хороший ген

    mutation_type = random.random()

    # Мутация преподавателя
    if mutation_type < 0.4:
        available_teachers = [t for t, subjs in teacher_subjects.items()
                              if subject.replace("лекция_", "") in subjs]
        if available_teachers:
            schedule[idx][0] = random.choice(available_teachers)

    # Мутация временного слота
    elif mutation_type < 0.8:
        current_day = lesson_slot.split('-')[0]
        available_slots = [ls for ls in LESSON_SLOTS
                           if ls.split('-')[0] == current_day and
                           not any(g[2] == group and g[1] == ls and g[4] == week_type
                                   for g in schedule if g != schedule[idx])]
        if available_slots:
            schedule[idx][1] = random.choice(available_slots)

    # Мутация типа недели
    else:
        schedule[idx][4] = "знаменатель" if week_type == "числитель" else "числитель"

    return schedule


def select_parent(ranked_population: List[Tuple[int, Schedule]]) -> Schedule:
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


def genetic_algorithm() -> Schedule:
    population = [generate_random_schedule() for _ in range(POPULATION_SIZE)]

    for generation in range(GENERATIONS):
        ranked = sorted([(calculate_fitness(ind), ind)
                         for ind in population], key=lambda x: x[0], reverse=True)

        elite = ranked[:int(POPULATION_SIZE * ELITISM_RATE)]
        survivors = ranked[:int(POPULATION_SIZE * SURVIVAL_RATE)]

        best_fitness, best_schedule = ranked[0]
        print(f"Generation {generation}, Best Fitness: {best_fitness}")

        if best_fitness >= 900_000:  # Достаточно хорошее решение
            print(f"Good solution found in generation {generation}")
            return best_schedule

        new_generation = [ind for (fit, ind) in elite]

        while len(new_generation) < POPULATION_SIZE:
            parent1 = select_parent(survivors)
            parent2 = select_parent(survivors)
            child = crossover(parent1, parent2)

            if random.random() < 0.1:
                child = mutate(child)

            new_generation.append(child)

        population = new_generation

    return max(population, key=calculate_fitness)


class ScheduleApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Генератор расписания университета")
        self.setGeometry(100, 100, 1000, 700)

        self.current_schedule = []
        self.week_type_filter = "все"

        self.init_ui()

    def init_ui(self):
        # Основные виджеты
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["Неделя", "День", "Время", "Группа", "Предмет", "Преподаватель"])
        self.table.setSortingEnabled(True)

        # Фильтры
        filter_layout = QVBoxLayout()

        week_type_combo = QComboBox()
        week_type_combo.addItems(["все", "числитель", "знаменатель"])
        week_type_combo.currentTextChanged.connect(self.filter_schedule)
        filter_layout.addWidget(QLabel("Фильтр по неделе:"))
        filter_layout.addWidget(week_type_combo)

        group_combo = QComboBox()
        group_combo.addItems(["все"] + groups)
        group_combo.currentTextChanged.connect(self.filter_schedule)
        filter_layout.addWidget(QLabel("Фильтр по группе:"))
        filter_layout.addWidget(group_combo)

        # Кнопки
        btn_layout = QVBoxLayout()
        self.generate_btn = QPushButton("Сгенерировать расписание")
        self.generate_btn.clicked.connect(self.generate_schedule)
        btn_layout.addWidget(self.generate_btn)

        self.save_btn = QPushButton("Сохранить в файл")
        self.save_btn.clicked.connect(self.save_schedule)
        btn_layout.addWidget(self.save_btn)

        self.load_btn = QPushButton("Загрузить из файла")
        self.load_btn.clicked.connect(self.load_schedule)
        btn_layout.addWidget(self.load_btn)

        # Основной layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(filter_layout)
        main_layout.addWidget(self.table)
        main_layout.addLayout(btn_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def generate_schedule(self):
        self.generate_btn.setEnabled(False)
        QApplication.processEvents()  # Обновляем интерфейс

        try:
            self.current_schedule = genetic_algorithm()
            self.display_schedule()
            QMessageBox.information(self, "Успех", "Расписание успешно сгенерировано!")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка генерации: {str(e)}")
        finally:
            self.generate_btn.setEnabled(True)

    def display_schedule(self, schedule=None):
        if schedule is None:
            schedule = self.current_schedule

        filtered = []
        for gene in schedule:
            week_type, slot, group, subject, teacher = gene[4], gene[1], gene[2], gene[3], gene[0]
            day, time = slot.split('-')

            if (self.week_type_filter == "все" or week_type == self.week_type_filter):
                filtered.append((week_type, day, time, group, subject, teacher))

        self.table.setRowCount(len(filtered))
        for row, (week, day, time, group, subject, teacher) in enumerate(filtered):
            self.table.setItem(row, 0, QTableWidgetItem(week))
            self.table.setItem(row, 1, QTableWidgetItem(day))
            self.table.setItem(row, 2, QTableWidgetItem(time))
            self.table.setItem(row, 3, QTableWidgetItem(group))
            self.table.setItem(row, 4, QTableWidgetItem(subject))
            self.table.setItem(row, 5, QTableWidgetItem(teacher))

        self.table.resizeColumnsToContents()

    def filter_schedule(self):
        if self.current_schedule:
            self.display_schedule()

    def save_schedule(self):
        if not self.current_schedule:
            QMessageBox.warning(self, "Ошибка", "Нет данных для сохранения")
            return

        try:
            data = []
            for gene in self.current_schedule:
                data.append({
                    "teacher": gene[0],
                    "slot": gene[1],
                    "group": gene[2],
                    "subject": gene[3],
                    "week_type": gene[4]
                })

            with open("schedule.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            QMessageBox.information(self, "Успех", "Расписание сохранено в schedule.json")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка сохранения: {str(e)}")

    def load_schedule(self):
        try:
            with open("schedule.json", "r", encoding="utf-8") as f:
                data = json.load(f)

            self.current_schedule = []
            for item in data:
                self.current_schedule.append([
                    item["teacher"],
                    item["slot"],
                    item["group"],
                    item["subject"],
                    item["week_type"]
                ])

            self.display_schedule()
            QMessageBox.information(self, "Успех", "Расписание загружено из schedule.json")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка загрузки: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ScheduleApp()
    window.show()
    sys.exit(app.exec_())