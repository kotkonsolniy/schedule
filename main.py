import random
from collections import defaultdict
from typing import List, Dict, Tuple
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTableWidget, QTableWidgetItem,
    QPushButton, QVBoxLayout, QWidget, QMessageBox, QFileDialog
)
from PyQt5.QtCore import Qt
import xlwt
import sys

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
subjects = ["интегралы", "япы", "физика", "джава", "физра"]
teachers = ["Иван", "Кирилл", "Варя", "Оля", "Коля"]

group_subject_requirements = {
    ("ИУ10-11", "интегралы"): 3,
    ("ИУ10-11", "физика"): 1,
    ("ИУ10-11", "физра"): 1,
    ("ИУ10-12", "интегралы"): 1,
    ("ИУ10-12", "япы"): 1,
    ("ИУ10-13", "джава"): 2,
    ("ИУ10-14", "япы"): 2,
    ("ИУ10-14", "физра"): 3,
    ("ИУ10-15", "интегралы"): 3,
    ("ИУ10-16", "физика"): 2,
    ("ИУ10-17", "джава"): 2,
    ("ИУ10-18", "физра"): 1,
    ("ИУ10-19", "япы"): 2
}

teacher_subjects = {
    "Иван": ["интегралы", "физика"],
    "Кирилл": ["джава"],
    "Варя": ["физра"],
    "Коля": ["япы"],
    "Оля": ["япы", "джава"]
}

lecture_groups = {
    "лекция_интегралы": ["ИУ10-11", "ИУ10-12", "ИУ10-15"],
    "лекция_физика": ["ИУ10-11", "ИУ10-13", "ИУ10-16"]
}


def generate_random_schedule() -> Schedule:
    schedule = []

    # Обычные занятия
    for (group, subject), count in group_subject_requirements.items():
        available_teachers = [t for t, subjs in teacher_subjects.items() if subject in subjs]
        if not available_teachers:
            continue

        for _ in range(count):
            gene = [
                random.choice(available_teachers),
                random.choice(LESSON_SLOTS),
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

        slot = random.choice(LESSON_SLOTS)
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


def calculate_fitness(schedule: Schedule) -> int:
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

        # Проверка преподавателя
        if subject.replace("лекция_", "") not in teacher_subjects.get(teacher, []):
            hard_constraints_violations += 1

        # Проверка двойного бронирования преподавателя
        if lesson_slot in teacher_lessons[teacher]:
            hard_constraints_violations += 1
        teacher_lessons[teacher].add(lesson_slot)

        # Проверка двойного бронирования группы
        if lesson_slot in group_lessons[group]:
            hard_constraints_violations += 1
        group_lessons[group].add(lesson_slot)

        # Подсчет пар в день
        day_lessons[day][group] += 1

        # Проверка на превышение лимита пар
        if day_lessons[day][group] > MAX_LESSONS_PER_DAY:
            hard_constraints_violations += 1

        # Подсчет окон
        slots = sorted([int(s.split('-')[1]) for s in group_lessons[group] if s.startswith(day)])
        for i in range(1, len(slots)):
            if slots[i] - slots[i - 1] > 1:
                windows_per_group[group] += 1

        subject_counts[(group, subject)] += 1

    # Проверка требований по предметам
    for (group, subject), required in group_subject_requirements.items():
        actual = subject_counts.get((group, subject), 0)
        hard_constraints_violations += abs(required - actual)

    # Штраф за окна
    hard_constraints_violations += sum(windows_per_group.values())

    return 1_000_000 - hard_constraints_violations * 10_000


def crossover(parent1: Schedule, parent2: Schedule) -> Schedule:
    child = []
    parent2_genes = parent2.copy()

    for gene in parent1:
        teacher, lesson_slot, group, subject = gene

        # Проверка конфликтов
        teacher_conflict = sum(1 for t, ls, g, s in parent1
                               if t == teacher and ls == lesson_slot) != 1
        group_conflict = sum(1 for t, ls, g, s in parent1
                             if g == group and ls == lesson_slot) != 1

        if not teacher_conflict and not group_conflict:
            child.append(gene)
        else:
            # Поиск альтернатив во втором родителе
            alternatives = [g for g in parent2_genes
                            if g[2] == group and g[3] == subject]

            # Фильтр по конфликтам
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


def mutate(schedule: Schedule) -> Schedule:
    if not schedule:
        return schedule

    idx = random.randint(0, len(schedule) - 1)
    teacher, lesson_slot, group, subject = schedule[idx]

    # Проверка текущего гена на конфликты
    teacher_conflict = sum(1 for t, ls, g, s in schedule
                           if t == teacher and ls == lesson_slot) != 1
    group_conflict = sum(1 for t, ls, g, s in schedule
                         if g == group and ls == lesson_slot) != 1

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
                           not any(g[2] == group and g[1] == ls
                                   for g in schedule if g != schedule[idx])]
        if available_slots:
            schedule[idx][1] = random.choice(available_slots)
        else:
            schedule[idx][1] = random.choice(LESSON_SLOTS)

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
        self.setGeometry(100, 100, 1200, 800)
        self.current_schedule = []
        self.init_ui()

    def init_ui(self):
        self.table = QTableWidget()
        self.table.setColumnCount(len(groups) + 2)  # +2 для дня и времени
        self.table.setHorizontalHeaderLabels(["День", "Время"] + groups)

        btn_layout = QVBoxLayout()
        self.generate_btn = QPushButton("Сгенерировать расписание")
        self.generate_btn.clicked.connect(self.generate_schedule)
        btn_layout.addWidget(self.generate_btn)

        self.save_btn = QPushButton("Сохранить в Excel")
        self.save_btn.clicked.connect(self.save_to_excel)
        btn_layout.addWidget(self.save_btn)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.table)
        main_layout.addLayout(btn_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def generate_schedule(self):
        self.generate_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            self.current_schedule = genetic_algorithm()
            self.display_schedule()
            QMessageBox.information(self, "Успех", "Расписание успешно сгенерировано!")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Ошибка генерации: {str(e)}")
        finally:
            self.generate_btn.setEnabled(True)

    def display_schedule(self):
        # Словарь для хранения данных: day -> time_num -> group -> subject
        schedule_data = {
            day: defaultdict(lambda: defaultdict(str))
            for day in DAYS
        }

        # Заполняем данные
        for teacher, lesson_slot, group, subject in self.current_schedule:
            day, time_num = lesson_slot.split('-')
            time_num = int(time_num)
            schedule_data[day][time_num][group] = f"{teacher}: {subject}"

        # Настраиваем таблицу
        total_rows = len(TIME_SLOTS) * len(DAYS)
        self.table.setRowCount(total_rows)
        self.table.setColumnCount(len(groups) + 2)

        row = 0
        for day in DAYS:
            # Записываем день только в первой строке для этого дня
            day_written = False

            for time_num in sorted(TIME_SLOTS.keys()):
                if not day_written:
                    # Создаем item для дня и устанавливаем row span
                    day_item = QTableWidgetItem(day)
                    self.table.setItem(row, 0, day_item)
                    self.table.setSpan(row, 0, 7, 1)  # Объединяем 7 строк
                    day_written = True

                self.table.setItem(row, 1, QTableWidgetItem(TIME_SLOTS[time_num]))

                for col, group in enumerate(groups, 2):
                    self.table.setItem(
                        row, col,
                        QTableWidgetItem(schedule_data[day][time_num].get(group, "---"))
                    )
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