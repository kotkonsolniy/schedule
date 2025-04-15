import random
from tabulate import tabulate
from collections import defaultdict
from typing import List, Tuple

# Типы для читаемости
Gene = List[str]
Schedule = List[Gene]

# Данные
groups = ["G1", "G2", "G3", "G4", "G5"]
subjects = ["S1", "S2", "S3", "S4", "S5"]
teachers = ["T1", "T2", "T3", "T4"]
lessons = ["L1", "L2", "L3", "L4", "L5", "L6"]

# Требуемые занятия: (группа, предмет) -> количество
d_subj = {
    ("G1", "S1"): 3, ("G1", "S3"): 1,
    ("G2", "S1"): 1, ("G2", "S2"): 1,
    ("G3", "S4"): 2,
    ("G4", "S2"): 2, ("G4", "S5"): 3,
    ("G5", "S1"): 3
}

# Кто какие предметы может преподавать
t_subj = {
    "T1": ["S1"],
    "T2": ["S2", "S3", "S4"],
    "T3": ["S1", "S3", "S5"],
    "T4": ["S2", "S5"]
}

POPULATION_SIZE = 250
GENERATIONS = 250


def generate_random_schedule() -> Schedule:
    schedule = []
    for (group, subject), count in d_subj.items():
        available_teachers = [t for t, subj in t_subj.items() if subject in subj]
        for _ in range(count):
            gene = [random.choice(available_teachers), random.choice(lessons), group, subject]
            schedule.append(gene)
    return schedule


def calc_fitness(schedule: Schedule) -> int:
    if len(schedule) != sum(d_subj.values()):
        return -10**6

    penalty = 0
    teacher_time = defaultdict(set)
    group_time = defaultdict(set)
    subject_counts = defaultdict(int)

    for teacher, lesson, group, subject in schedule:
        # Преподаватель в одно время — только на одном занятии
        if lesson in teacher_time[teacher]:
            penalty += 1
        teacher_time[teacher].add(lesson)

        # Группа в одно время — только на одном занятии
        if lesson in group_time[group]:
            penalty += 1
        group_time[group].add(lesson)

        # Предмет должен преподавать компетентный преподаватель
        if subject not in t_subj[teacher]:
            penalty += 1

        subject_counts[(group, subject)] += 1

    # Расхождение с требованиями по предметам
    for key, required in d_subj.items():
        actual = subject_counts.get(key, 0)
        penalty += abs(required - actual)

    # Избыточные занятия
    for key in subject_counts:
        if key not in d_subj:
            penalty += subject_counts[key]

    return 1_000_000 - penalty * 10_000


def crossover(parent1: Schedule, parent2: Schedule) -> Schedule:
    child = []
    for gene in parent1:
        teacher, lesson, group, subject = gene
        if is_slot_safe(child, teacher, lesson, group):
            child.append(gene)
        else:
            # Поищем альтернативу из parent2
            variants = [g for g in parent2 if g[2] == group and g[3] == subject]
            safe_variants = [v for v in variants if is_slot_safe(child, v[0], v[1], v[2])]
            child.append(random.choice(safe_variants if safe_variants else variants))
    return child


def is_slot_safe(schedule: Schedule, teacher: str, lesson: str, group: str) -> bool:
    for t, l, g, _ in schedule:
        if (t == teacher and l == lesson) or (g == group and l == lesson):
            return False
    return True


def mutate(schedule: Schedule) -> Schedule:
    idx = random.randint(0, len(schedule) - 1)
    teacher, lesson, group, subject = schedule[idx]

    available_teachers = [t for t, subs in t_subj.items() if subject in subs]
    unavailable_lessons = {l for t, l2, g, s in schedule if g == group}
    available_lessons = [l for l in lessons if l not in unavailable_lessons]

    if random.random() > 0.5:
        schedule[idx][0] = random.choice(available_teachers)
    elif available_lessons:
        schedule[idx][1] = random.choice(available_lessons)
    else:
        schedule[idx][1] = random.choice(lessons)

    return schedule


def selection(ranked_population: List[Tuple[int, Schedule]]) -> Schedule:
    total_fitness = sum(max(f, 0) for f, _ in ranked_population)
    if total_fitness == 0:
        return random.choice(ranked_population)[1]

    pick = random.uniform(0, total_fitness)
    current = 0
    for fitness, individual in ranked_population:
        current += max(fitness, 0)
        if current >= pick:
            return individual
    return ranked_population[0][1]


def visualize(schedule: Schedule) -> None:
    time_table = defaultdict(lambda: defaultdict(list))
    time_slots = sorted({lesson[1] for lesson in schedule})
    group_list = sorted({lesson[2] for lesson in schedule})

    for teacher, lesson, group, subject in schedule:
        time_table[lesson][group].append(f"{teacher}: {subject}")

    headers = ["Время"] + group_list
    table = []

    for lesson in time_slots:
        row = [lesson]
        for group in group_list:
            entries = time_table[lesson].get(group, [])
            if len(entries) > 1:
                row.append("КОНФЛИКТ!\n" + "\n".join(entries))
            elif entries:
                row.append("\n".join(entries))
            else:
                row.append("---")
        table.append(row)

    print(tabulate(table, headers=headers, tablefmt="grid", stralign="left"))
    print("\nПримечания:")
    print("- КОНФЛИКТ! означает несколько занятий одновременно")
    print("- --- означает отсутствие занятий")
    print("Приспособленность:", calc_fitness(schedule))
    print("-----------------")


def genetic_algorithm() -> Schedule:
    population = [generate_random_schedule() for _ in range(POPULATION_SIZE)]

    for generation in range(GENERATIONS):
        ranked = sorted([(calc_fitness(ind), ind) for ind in population], reverse=True)
        elite = [ind for _, ind in ranked[:POPULATION_SIZE // 5]]
        alive = ranked[:int(POPULATION_SIZE * 0.8)]

        best = ranked[0][1]
        visualize(best)
        if calc_fitness(best) == 1_000_000:
            print(f"Идеальное расписание найдено на поколении {generation}")
            break

        new_population = elite.copy()
        while len(new_population) < POPULATION_SIZE:
            parent1 = selection(alive)
            parent2 = selection(alive)
            child = crossover(parent1, parent2)
            new_population.append(mutate(child))

        population = new_population

    return max(population, key=calc_fitness)


# Запуск
visualize(genetic_algorithm())
