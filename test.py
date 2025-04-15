import random
from tabulate import tabulate # для красивого отображения таблицы в консоли
from collections import defaultdict #для подсчета задания и выдачи расписани (для новых ключей)
from typing import List, Dict, Tuple, Set #для типов данных

# Constants
POPULATION_SIZE = 250 #>повышает разнообразитие решений
GENERATIONS = 250 #макс число поколений >шанс найти гут решение
ELITISM_RATE = 0.2 #отбирает 20 процентов лучших расписаний элита
SURVIVAL_RATE = 0.8 #от остальных 80 берется лучшее и поллучаем также 250 расписаний

# Data structures, типы групп, определение типо в gene and shedule
Group = str
Subject = str
Teacher = str
LessonSlot = str
Gene = List  # [Teacher, LessonSlot, Group, Subject]
Schedule = List[Gene]

# Наши базы данных
groups = ["ИУ10-11", "ИУ10-12", "ИУ10-13", "ИУ10-14", "ИУ10-15"]
subjects = ["интегралы", "япы", "физика", "джава", "физра"]
teachers = ["Иван", "Кирилл", "Варя", "Оля", "Коля"]  # Добавили Колю
lesson_slots = ["8-9", "9-10", "10-11", "11-12", "12-13", "13-14"]

# Связь групп с предметами
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

# Связь преподавателей с предметами
teacher_subjects = {
    "Иван": ["интегралы", "физика"],  # Добавили физику
    "Кирилл": ["джава"],
    "Варя": ["физра"],
    "Коля": ["япы"],
    }

#создаёт случайное расписание занятий на основе заданных требований
def generate_random_schedule() -> Schedule:
    schedule = []
    for (group, subject), count in group_subject_requirements.items(): #перебор требований к занятиям
        available_teachers = [t for t, subjs in teacher_subjects.items() if subject in subjs]
        if not available_teachers:
            raise ValueError(f"Нет доступных учителей по предмету {subject}")

        for _ in range(count): #ген-занятие создает ген, добавлляет в расписание
            gene = [
                random.choice(available_teachers),
                random.choice(lesson_slots),
                group,
                subject
            ]
            schedule.append(gene)
    return schedule

#оценивает качество расписания, возвращаеь оценку расписанию отри знач неприемлемо 1000000 отл
def calculate_fitness(schedule: Schedule) -> int:
    if len(schedule) != sum(group_subject_requirements.values()): #проверкаколлва занятий жест огр
        return -1_000_000

    # Преподаватель не ведёт данный предмет
    # Преподаватель в два места одновременно
    # Группа на два занятия одновременно
    # Несоответствие количества занятий требованиям
    # Наличие лишних занятий

    hard_constraints_violations = 0
    teacher_lessons = defaultdict(set)  # {teacher: {lesson_slots}}
    group_lessons = defaultdict(set)  # {group: {lesson_slots}}
    subject_counts = defaultdict(int)  # {(group, subject): count}

    for teacher, lesson_slot, group, subject in schedule:
        # Check teacher can teach this subject
        if subject not in teacher_subjects[teacher]:
            hard_constraints_violations += 1

        # Check teacher isn't double-booked
        if lesson_slot in teacher_lessons[teacher]:
            hard_constraints_violations += 1
        teacher_lessons[teacher].add(lesson_slot)

        # Check group isn't double-booked
        if lesson_slot in group_lessons[group]:
            hard_constraints_violations += 1
        group_lessons[group].add(lesson_slot)

        # Count subject occurrences per group
        subject_counts[(group, subject)] += 1

    # Check subject requirements are met
    for (group, subject), required in group_subject_requirements.items():
        actual = subject_counts.get((group, subject), 0)
        hard_constraints_violations += abs(required - actual)

    # Check no extra subjects are scheduled
    for (group, subject), actual in subject_counts.items():
        if (group, subject) not in group_subject_requirements:
            hard_constraints_violations += actual

    return 1_000_000 - hard_constraints_violations * 10_000


def crossover(parent1: Schedule, parent2: Schedule) -> Schedule:
    #Создание расписания для ребенка, объединив двух родителей
    child = []
    for gene in parent1: #обход генов лучшего родителя
        teacher, lesson_slot, group, subject = gene

        # Check if the gene is conflict-free in parent1
        teacher_conflict = sum(1 for t, l, g, s in parent1
                               if t == teacher and l == lesson_slot) != 1
        group_conflict = sum(1 for t, l, g, s in parent1
                             if g == group and l == lesson_slot) != 1

        if not teacher_conflict and not group_conflict:
            child.append(gene)
        else:
            # Try to find a better gene from parent2
            alternatives = [g for g in parent2 #поиск альтернатив у 2го родителя
                            if g[2] == group and g[3] == subject]

            # Filter conflict-free alternatives выбирает без конфликтов или случайное если все конфликтны
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
                child.append(random.choice(valid_alternatives))
            else:
                child.append(random.choice(alternatives))

    return child


def mutate(schedule: Schedule) -> Schedule:
    #Случайно мутировать расписание, чтобы внести разнообразие
    idx = random.randint(0, len(schedule) - 1)#вносит случайные изменения в одно случайно выбранное занятие(ген)
    teacher, lesson_slot, group, subject = schedule[idx]

    # Check if the gene is already noconflict
    teacher_conflict = sum(1 for t, l, g, s in schedule #два занятиу перепода одновременно
                           if t == teacher and l == lesson_slot) != 1
    group_conflict = sum(1 for t, l, g, s in schedule #два занятиу у группы одновременно
                         if g == group and l == lesson_slot) != 1

    if not teacher_conflict and not group_conflict:
        return schedule  # No need to mutate

    # Mutation options
    mutation_type = random.random()

    # Mutate teacher при отсутсвии конфликтов происходит мутация с вероятностью в 50 процентов
    if mutation_type < 0.5:
        available_teachers = [t for t, subjs in teacher_subjects.items()
                              if subject in subjs]
        schedule[idx][0] = random.choice(available_teachers)

    # Mutate lesson slot
    else:
        # Get available slots where group has no lesson
        group_lessons = {l for t, l, g, s in schedule if g == group}#случайн мутация по смегне препода
        available_slots = [l for l in lesson_slots if l not in group_lessons] #случайная смена слота

        if available_slots:
            schedule[idx][1] = random.choice(available_slots)
        else:
            schedule[idx][1] = random.choice(lesson_slots)
#мутирует прогблемные гены, случайность без конфликтов, модифицирует занятия
    return schedule


#выбирает одно расписание для участия в скрещивании (чем онолучше тем больше вероятность быть выбранным)
def select_parent(ranked_population: List[Tuple[int, Schedule]]) -> Schedule:
    #Выберите родителя, используя выбор колеса рулетки
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

#чисто визуалка
def visualize_schedule(schedule: Schedule):
    """Display the schedule in a readable table format."""
    time_schedule = defaultdict(lambda: defaultdict(list))
    used_slots = {lesson[1] for lesson in schedule}

    for teacher, slot, group, subject in schedule:
        entry = f"{teacher}: {subject}"
        time_schedule[slot][group].append(entry)

    # Prepare table data
    table = []
    headers = ["Time"] + groups

    for slot in sorted(used_slots):
        row = [slot]
        for group in groups:
            lessons = time_schedule[slot].get(group, [])

            if len(lessons) > 1:
                row.append("CONFLICT!\n" + "\n".join(lessons))
            elif lessons:
                row.append("\n".join(lessons))
            else:
                row.append("---")
        table.append(row)

    print(tabulate(table, headers=headers, tablefmt="grid", stralign="left"))
    print("\nNotes:")
    print("- CONFLICT! means multiple lessons at the same time")
    print("- --- means no lesson scheduled")
    print(f"Fitness score: {calculate_fitness(schedule)}")
    print("-----------------")


#реализует эволюционный процесс, где расписания (особи) постепенно улучшаются через несколько поколений
def genetic_algorithm() -> Schedule:
    population = [generate_random_schedule() for _ in range(POPULATION_SIZE)]

    for generation in range(GENERATIONS):
        # Оценить и ранжировать популяцию
        ranked = sorted([(calculate_fitness(ind), ind)
                         for ind in population], reverse=True)

        # Удерживать лучших людей (элитарность)
        elite = ranked[:int(POPULATION_SIZE * ELITISM_RATE)]
        survivors = ranked[:int(POPULATION_SIZE * SURVIVAL_RATE)]

        # показ ллучшего расписания поколения
        best_fitness, best_schedule = max(ranked, key=lambda x: x[0])
        print(f"Generation {generation}, Best Fitness: {best_fitness}")
        visualize_schedule(best_schedule)

        if best_fitness == 1_000_000:
            print(f"Perfect solution found in generation {generation}")
            return best_schedule

        # Create new generation
        new_generation = [ind for (fit, ind) in elite]
#скрещивание и мутации
        while len(new_generation) < POPULATION_SIZE:
            parent1 = select_parent(survivors)
            parent2 = select_parent(survivors)
            child = crossover(parent1, parent2)

            if random.random() < 0.1:  # Mutation probability
                child = mutate(child)

            new_generation.append(child)

        population = new_generation

    return max(population, key=calculate_fitness)


if __name__ == "__main__":
    try:
        best_schedule = genetic_algorithm()
        print("\nФинальное лучшее расписание:")
        visualize_schedule(best_schedule)
    except ValueError as e:
        print(f"Ошибка: {e}")
        print("Пожалуйста, проверьте введенные вами данные. - убедитесь, что по всем предметам назначен хотя бы один учитель.")
