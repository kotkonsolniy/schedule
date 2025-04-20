import random
from collections import defaultdict
from typing import List, Dict, Tuple, Set
from tabulate import tabulate

# Constants
POPULATION_SIZE = 250
GENERATIONS = 250
ELITISM_RATE = 0.2
SURVIVAL_RATE = 0.8
MUTATION_RATE = 0.1

# Data types
Group = str
Subject = str
Teacher = str
Classroom = str
LessonSlot = str
Day = str
Gene = List  # [Teacher, LessonSlot, Group, Subject, Classroom, Day]
Schedule = List[Gene]

# Sample data
groups = ["ИУ10-11", "ИУ10-12", "ИУ10-13", "ИУ10-14", "ИУ10-15"]
subjects = ["интегралы", "япы", "физика", "джава", "физра"]
teachers = ["Иван", "Кирилл", "Варя", "Оля", "Коля"]
days = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб"]
lesson_slots = ["1", "2", "3", "4", "5", "6"]

# Classrooms with their features
classrooms = {
    "А-101": {"seats": 30, "projector": True, "computers_win": 10, "acoustics": True},
    "А-202": {"seats": 25, "projector": False, "computers_linux": 15, "soldering": True},
    "Б-103": {"seats": 50, "projector": True, "sound_system": True, "emsc": True},
    "Г-304": {"seats": 20, "projector": True, "computers_win": 20, "spectrum_analysis": True},
    "Д-105": {"seats": 40, "projector": True, "acoustics": True, "tempest": True}
}

# Subject requirements for classrooms
subject_classroom_requirements = {
    "физика": {"projector": True, "seats": 25},
    "джава": {"computers_win": True, "seats": 20},
    "япы": {"computers_linux": True, "seats": 15},
    "физра": {"sound_system": True, "seats": 30},
    "интегралы": {"projector": True, "seats": 20}
}

# Group-subject requirements
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

# Teacher-subject assignments
teacher_subjects = {
    "Иван": ["интегралы", "физика"],
    "Кирилл": ["джава"],
    "Варя": ["физра"],
    "Коля": ["япы"],
}


def generate_random_schedule() -> Schedule:
    schedule = []
    for (group, subject), count in group_subject_requirements.items():
        available_teachers = [t for t, subjs in teacher_subjects.items() if subject in subjs]
        if not available_teachers:
            raise ValueError(f"No available teachers for subject {subject}")

        # Filter classrooms that meet subject requirements
        reqs = subject_classroom_requirements.get(subject, {})
        available_classrooms = [
            room for room, features in classrooms.items()
            if all(features.get(k, 0) >= (v if isinstance(v, int) else 1) for k, v in reqs.items())
        ]

        if not available_classrooms:
            raise ValueError(f"No suitable classrooms for subject {subject}")

        for _ in range(count):
            gene = [
                random.choice(available_teachers),
                random.choice(lesson_slots),
                group,
                subject,
                random.choice(available_classrooms),
                random.choice(days)
            ]
            schedule.append(gene)
    return schedule


def calculate_fitness(schedule: Schedule) -> int:
    if len(schedule) != sum(group_subject_requirements.values()):
        return -1_000_000

    hard_constraints_violations = 0
    soft_constraints_violations = 0

    # Trackers for constraints
    teacher_lessons = defaultdict(set)  # {(teacher, day): {slot}}
    group_lessons = defaultdict(set)  # {(group, day): {slot}}
    classroom_lessons = defaultdict(set)  # {(classroom, day): {slot}}
    teacher_days = defaultdict(set)  # {teacher: {days}}
    group_days = defaultdict(set)  # {group: {days}}
    subject_counts = defaultdict(int)  # {(group, subject): count}
    teacher_work_days = defaultdict(int)  # {teacher: days_count}
    group_work_days = defaultdict(int)  # {group: days_count}

    for teacher, slot, group, subject, classroom, day in schedule:
        # Hard constraints
        # 1. Teacher can teach this subject
        if subject not in teacher_subjects[teacher]:
            hard_constraints_violations += 1

        # 2. Teacher isn't double-booked
        if (teacher, day, slot) in teacher_lessons:
            hard_constraints_violations += 1
        teacher_lessons[(teacher, day)].add(slot)

        # 3. Group isn't double-booked
        if (group, day, slot) in group_lessons:
            hard_constraints_violations += 1
        group_lessons[(group, day)].add(slot)

        # 4. Classroom isn't double-booked
        if (classroom, day, slot) in classroom_lessons:
            hard_constraints_violations += 1
        classroom_lessons[(classroom, day)].add(slot)

        # 5. Classroom meets requirements
        reqs = subject_classroom_requirements.get(subject, {})
        room_features = classrooms[classroom]
        if not all(room_features.get(k, 0) >= (v if isinstance(v, int) else 1) for k, v in reqs.items()):
            hard_constraints_violations += 1

        # 6. Subject requirements are met
        subject_counts[(group, subject)] += 1

        # Track days for teachers and groups
        teacher_days[teacher].add(day)
        group_days[group].add(day)

    # Check subject requirements
    for (group, subject), required in group_subject_requirements.items():
        actual = subject_counts.get((group, subject), 0)
        hard_constraints_violations += abs(required - actual)

    # Check no extra subjects
    for (group, subject), actual in subject_counts.items():
        if (group, subject) not in group_subject_requirements:
            hard_constraints_violations += actual

    # Soft constraints
    # 1. No more than 6 lessons per day for group
    for (group, day), slots in group_lessons.items():
        if len(slots) > 6:
            soft_constraints_violations += 1

    # 2. No more than 1 window per day for group
    for (group, day), slots in group_lessons.items():
        slots_sorted = sorted(int(s) for s in slots)
        windows = 0
        for i in range(1, len(slots_sorted)):
            if slots_sorted[i] - slots_sorted[i - 1] > 1:
                windows += 1
        if windows > 1:
            soft_constraints_violations += 1

    # 3. No window before or after lunch (assuming lunch is after 3rd slot)
    for (group, day), slots in group_lessons.items():
        slots_int = [int(s) for s in slots]
        if 3 in slots_int:
            if (2 in slots_int and 4 not in slots_int) or (4 in slots_int and 2 not in slots_int):
                soft_constraints_violations += 1

    # 4. Teacher's lessons are compact (same day)
    for teacher, days_set in teacher_days.items():
        if len(days_set) > 1:
            # Penalize for each extra day
            soft_constraints_violations += len(days_set) - 1

    # 5. Minimize transitions between buildings (first letter of classroom)
    building_transitions = 0
    for group in groups:
        group_classes = [g for g in schedule if g[2] == group]
        buildings = [c[4][0] for c in group_classes]  # First letter of classroom
        for i in range(1, len(buildings)):
            if buildings[i] != buildings[i - 1]:
                building_transitions += 1

    soft_constraints_violations += building_transitions * 0.5  # Smaller penalty

    # 6. Optimal use of weekends (prefer weekdays)
    weekend_classes = sum(1 for gene in schedule if gene[5] == "Сб")
    soft_constraints_violations += weekend_classes * 0.2

    # Calculate total fitness
    if hard_constraints_violations > 0:
        return -1_000_000 - hard_constraints_violations * 10_000

    return 1_000_000 - soft_constraints_violations * 100


def crossover(parent1: Schedule, parent2: Schedule) -> Schedule:
    child = []
    for gene in parent1:
        teacher, lesson_slot, group, subject, classroom, day = gene

        # Check if gene is conflict-free in parent1
        teacher_conflict = sum(1 for t, l, g, s, c, d in parent1
                               if t == teacher and d == day and l == lesson_slot) != 1
        group_conflict = sum(1 for t, l, g, s, c, d in parent1
                             if g == group and d == day and l == lesson_slot) != 1
        classroom_conflict = sum(1 for t, l, g, s, c, d in parent1
                                 if c == classroom and d == day and l == lesson_slot) != 1

        if not teacher_conflict and not group_conflict and not classroom_conflict:
            child.append(gene)
        else:
            # Try to find better gene from parent2
            alternatives = [g for g in parent2
                            if g[2] == group and g[3] == subject]

            # Filter conflict-free alternatives
            valid_alternatives = []
            for alt in alternatives:
                alt_teacher, alt_lesson, alt_group, alt_subject, alt_classroom, alt_day = alt
                teacher_ok = sum(1 for t, l, g, s, c, d in child
                                 if t == alt_teacher and d == alt_day and l == alt_lesson) == 0
                group_ok = sum(1 for t, l, g, s, c, d in child
                               if g == alt_group and d == alt_day and l == alt_lesson) == 0
                classroom_ok = sum(1 for t, l, g, s, c, d in child
                                   if c == alt_classroom and d == alt_day and l == alt_lesson) == 0

                if teacher_ok and group_ok and classroom_ok:
                    valid_alternatives.append(alt)

            if valid_alternatives:
                child.append(random.choice(valid_alternatives))
            else:
                child.append(random.choice(alternatives))

    return child


def mutate(schedule: Schedule) -> Schedule:
    idx = random.randint(0, len(schedule) - 1)
    teacher, lesson_slot, group, subject, classroom, day = schedule[idx]

    # Check if gene is already conflict-free
    teacher_conflict = sum(1 for t, l, g, s, c, d in schedule
                           if t == teacher and d == day and l == lesson_slot) != 1
    group_conflict = sum(1 for t, l, g, s, c, d in schedule
                         if g == group and d == day and l == lesson_slot) != 1
    classroom_conflict = sum(1 for t, l, g, s, c, d in schedule
                             if c == classroom and d == day and l == lesson_slot) != 1

    if not teacher_conflict and not group_conflict and not classroom_conflict:
        if random.random() < 0.3:  # Small chance to mutate anyway
            mutation_type = random.random()
            if mutation_type < 0.3:
                # Mutate classroom
                reqs = subject_classroom_requirements.get(subject, {})
                available_classrooms = [
                    room for room, features in classrooms.items()
                    if all(features.get(k, 0) >= (v if isinstance(v, int) else 1)
                           for k, v in reqs.items())
                ]
                if available_classrooms:
                    schedule[idx][4] = random.choice(available_classrooms)
                elif mutation_type < 0.6:
                # Mutate day
                    schedule[idx][5] = random.choice(days)
                else:
                # Mutate slot
                    schedule[idx][1] = random.choice(lesson_slots)
        return schedule

    # Mutation options
    mutation_type = random.random()

    if mutation_type < 0.4:  # Mutate classroom
        reqs = subject_classroom_requirements.get(subject, {})
        available_classrooms = [
            room for room, features in classrooms.items()
            if all(features.get(k, 0) >= (v if isinstance(v, int) else 1)
                   for k, v in reqs.items())
        ]
        if available_classrooms:
            schedule[idx][4] = random.choice(available_classrooms)

    elif mutation_type < 0.7:  # Mutate day
        schedule[idx][5] = random.choice(days)

    else:  # Mutate lesson slot
        # Get available slots where group has no lesson that day
        group_day_lessons = {l for t, l, g, s, c, d in schedule
                             if g == group and d == day}
        available_slots = [l for l in lesson_slots if l not in group_day_lessons]

        if available_slots:
            schedule[idx][1] = random.choice(available_slots)
        else:
            schedule[idx][1] = random.choice(lesson_slots)

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


def visualize_schedule(schedule: Schedule):
    """Display the schedule in a readable table format."""
    day_schedule = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for teacher, slot, group, subject, classroom, day in schedule:
        entry = f"{teacher}: {subject} ({classroom})"
        day_schedule[day][slot][group].append(entry)

    # Prepare table data for each day
    for day in days:
        print(f"\n{day}:")
        time_schedule = day_schedule[day]
        if not time_schedule:
            print("  No classes scheduled")
            continue

        table = []
        headers = ["Time"] + groups

        for slot in sorted(time_schedule.keys(), key=int):
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


def genetic_algorithm() -> Schedule:
    population = [generate_random_schedule() for _ in range(POPULATION_SIZE)]

    for generation in range(GENERATIONS):
        ranked = sorted([(calculate_fitness(ind), ind)
                         for ind in population], reverse=True)

        elite = ranked[:int(POPULATION_SIZE * ELITISM_RATE)]
        survivors = ranked[:int(POPULATION_SIZE * SURVIVAL_RATE)]

        best_fitness, best_schedule = max(ranked, key=lambda x: x[0])
        print(f"Generation {generation}, Best Fitness: {best_fitness}")

        if generation % 50 == 0:  # Show progress every 50 generations
            visualize_schedule(best_schedule)

        if best_fitness == 1_000_000:
            print(f"Perfect solution found in generation {generation}")
            return best_schedule

        new_generation = [ind for (fit, ind) in elite]

        while len(new_generation) < POPULATION_SIZE:
            parent1 = select_parent(survivors)
            parent2 = select_parent(survivors)
            child = crossover(parent1, parent2)

            if random.random() < MUTATION_RATE:
                child = mutate(child)

            new_generation.append(child)

        population = new_generation

    return max(population, key=calculate_fitness)


if __name__ == "__main__":
    try:
        best_schedule = genetic_algorithm()
        print("\nFinal best schedule:")
        visualize_schedule(best_schedule)
    except ValueError as e:
        print(f"Error: {e}")
        print("Please check your input data - make sure all subjects have at least one teacher assigned.")