import random
from tabulate import tabulate
from collections import defaultdict

def crossover(parent1, parent2):
    child = []
    for gene in parent1:
        if len([[t,l,g,s] for t, l, g, s in parent1 if t== gene[0] and l== gene[1]]) == 1 and len([[t,l,g,s] for t, l, g, s in parent1 if g== gene[2] and l== gene[1]]) == 1 :
            child.append(gene)
        else:
            group = gene[2]
            subj = gene[3]
            variants = [[t,l,g,s] for t, l, g, s in parent2 if s== subj and g== group]
            a_var = [variant for variant in variants if len([[t,l,g,s] for t, l, g, s in child if t== variant[0] and l== variant[1]]) == 1 and len([[t,l,g,s] for t, l, g, s in child if g== variant[2] and l== variant[1]]) == 1 ]
            if len(a_var) > 0:
                child.append(random.choice(a_var))
            else:
                child.append(random.choice(variants))
    return child

def mutate(schedule):
    
    idx = random.randint(0, len(schedule)-1)
    gene = schedule[idx]
    if len([[t,l,g,s] for t, l, g, s in schedule if t== gene[0] and l== gene[1]]) == 1 and len([[t,l,g,s] for t, l, g, s in schedule if g== gene[2] and l== gene[1]]) == 1 :
       pass
    subj = gene[3]
    teach_avail = [t for t, s in t_subj.items() if subj in s]
    l_avail = [l for l in lessons if l not in [lg for t, lg, g, s in schedule if g== gene[2]]]
    if random.random() > random.random():
        schedule[idx][0] = random.choice(teach_avail)
    elif len(l_avail) == 0:
        schedule[idx][1] = random.choice(lessons)
    else:
        schedule[idx][1] = random.choice(l_avail)
    return schedule

POPULATION_SIZE = 250
GENERATIONS = 250


def vt(schedule):
    time_schedule = defaultdict(lambda: defaultdict(list))
    time_slots = sorted({lesson[1] for lesson in schedule})
    groups = sorted({lesson[2] for lesson in schedule})
    for lesson in schedule:
        teacher, time, group, subject = lesson
        entry = f"{teacher}: {subject}"
        time_schedule[time][group].append(entry)

    table = []
    headers = ["Время"] + groups
    
    for time in time_slots:
        row = [time]
        for group in groups:
            lessons = time_schedule[time].get(group, [])
            
            if len(lessons) > 1:
                row.append("КОНФЛИКТ!\n" + "\n".join(lessons))
            elif lessons:
                row.append("\n".join(lessons))
            else:
                row.append("---")
        table.append(row)

    print(tabulate(table, headers=headers, tablefmt="grid", stralign="left"))
    print("\nПримечания:")
    print("- КОНФЛИКТ! означает несколько занятий одновременно")
    print("- --- означает отсутствие занятий")
    print(calcfit(schedule))
    print("-----------------")

groups = ["G1","G2", "G3", "G4","G5"]
subjects = ["S1", "S2", "S3", "S4", "S5"]
teachers = ["T1","T2","T3", "T4"]
lessons = ["L1","L2","L3","L4","L5"]

d_subj={
("G1", "S1"):3,
("G1", "S3"):1, 
("G2","S1"):1,
("G2","S2"):1,
("G3","S4"):2,
("G4","S2"):2,
("G4","S5"):3,
("G5","S1"):3
}
t_subj={
"T1":["S1"], 
"T2":["S2","S3","S4"], 
"T3":["S1","S3","S5"], 
"T4":["S2","S5"]
}

def genrnd():
    chromosome = []
    required = {k: v for k, v in d_subj.items()}  # Копируем требования
    
    for (gs,count) in d_subj.items():
        (group, subj) = gs
        teach_avail = [t for t, s in t_subj.items() if subj in s]
       
        for _ in range(count):
            chromosome.append([random.choice(teach_avail), random.choice(lessons), group, subj])
    return chromosome

def selection(ranked_population):
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



def calcfit(chromosome):
    if len(chromosome) != sum([count for gr_less, count in d_subj.items()]):
        return -1000000 
  
    fitness = 0
    teacher_lessons = defaultdict(set)
    group_lessons = defaultdict(set)
    subject_counts = defaultdict(int)
    hard_constraints = 0  

    for gene in chromosome:
        t, l, g, s = gene

        if len([[t,l,g,s] for t, l, g, s in chromosome if t== gene[0] and l== gene[1]]) != 1:
            hard_constraints += 1
        if len([[t,l,g,s] for t, l, g, s in chromosome if g== gene[2] and l== gene[1]]) != 1 :
            hard_constraints += 1

        if s not in t_subj[t]:
            hard_constraints += 1
            
        teacher_lessons[t].add(l)
       
        group_lessons[g].add(l)
        
        subject_counts[(g, s)] += 1
    
    for (g, s), required in d_subj.items():
        actual = subject_counts.get((g, s), 0)
        hard_constraints += abs(required - actual)
        
    for (g, s), actual in subject_counts.items():      
        if (g, s) not in d_subj:
            hard_constraints += actual

    return (1000000 - hard_constraints*10000)



def g_alg():
    
    population = [genrnd() for _ in range(POPULATION_SIZE)]
    
    for generation in range(GENERATIONS):
       
        ranked = sorted([(calcfit(ind), ind) 
                      for ind in population], reverse=True)
        
        elite = ranked[:int(POPULATION_SIZE*0.2)] 
        alive = ranked[:int(POPULATION_SIZE*0.8)]

        best = max(population, key=calcfit)
        vt(best)
        if calcfit(best) == 1000000:
            print("end on genetation", generation)
            break
        print ("-----------------")

        new_generation = [ind for (fit, ind) in elite]
        
        while len(new_generation) < POPULATION_SIZE:
            parent1 = selection(alive)
            parent2 = selection(alive)
            child = crossover(parent1, parent2)
           
            child = mutate(child)
            new_generation.append(child)
        
        population = new_generation
    
    return max(population, key=calcfit)

vt(g_alg())

